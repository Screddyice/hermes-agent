"""Regression tests for approval prompt credential redaction (issue #48456).

When Tirith flags a command for containing a credential-shaped pattern, the
gateway approval prompt must redact the credential from the command text
before sending it to the chat platform. Without this fix, the raw command
(with the credential in plaintext) is sent verbatim to Telegram/Discord/etc.,
undoing Tirith's redaction one layer up.

The tests exercise the module-level ``_redact_approval_command`` seam and both
approval transports at runtime. They fail if either the chat-platform prompt
or the Runs API response exposes the original command.

Credential fixtures are built at runtime from a benign prefix + a run of
``X`` characters (the same trick tests/agent/test_redact.py uses): they match
the redactor regexes so the assertions stay meaningful, but contain no real
or real-looking key, so secret scanners do not flag this file.
"""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import TurnRunner, _redact_approval_command
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tools import approval as approval_mod

# Synthetic, scanner-safe credential fixtures. Each matches its redactor
# regex (ghp_/sk-/JWT) but is unmistakably fake -- a run of X's, never a
# real or real-format key.
_FAKE_GHP = "ghp_" + "X" * 36
_FAKE_OPENAI = "sk-proj-" + "X" * 40
_FAKE_JWT = "eyJ" + "X" * 20 + "." + "eyJ" + "X" * 24 + "." + "X" * 30


class TestRedactApprovalCommand:
    """Contract for the approval-prompt redaction seam used by the gateway."""

    def test_redacts_github_pat(self):
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com/user"
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out
        # command structure preserved so the operator can still judge the action
        assert "curl" in out
        assert "github.com" in out

    def test_redacts_openai_key(self):
        raw = "export OPENAI_API_KEY=" + _FAKE_OPENAI + " && python s.py"
        out = _redact_approval_command(raw)
        assert _FAKE_OPENAI not in out
        assert "python s.py" in out

    def test_redacts_bearer_token(self):
        raw = "curl -H 'Authorization: Bearer " + _FAKE_JWT + "' https://api.example.com"
        out = _redact_approval_command(raw)
        assert _FAKE_JWT not in out


    def test_forces_redaction_even_when_disabled(self, monkeypatch):
        """force=True must redact even if security.redact_secrets is off -- the
        approval prompt is a hard secret-egress boundary regardless of config."""
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com"
        # With redaction globally disabled, the seam must STILL redact (force=True).
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False, raising=False)
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out


class TestApprovalCommandWiring:
    """Exercise both approval-notify transports through their public effects."""

    def test_chat_platform_path_redacts_before_send(self):
        sent = {}

        class _ApprovalAdapter:
            def pause_typing_for_chat(self, _chat_id):
                return None

            async def send_exec_approval(self, **kwargs):
                sent.update(kwargs)
                return SimpleNamespace(success=True, error=None)

        class _ApprovalAgent:
            def __init__(self, **kwargs):
                self.model = kwargs["model"]
                self.session_id = kwargs["session_id"]
                self.tools = []
                self.context_compressor = SimpleNamespace(
                    last_prompt_tokens=0,
                    context_length=200_000,
                )
                self.session_prompt_tokens = 0
                self.session_completion_tokens = 0

            def run_conversation(self, _message, **_kwargs):
                notify = approval_mod._gateway_notify_cbs["approval-redaction-session"]
                notify({
                    "command": "curl -H 'Authorization: token " + _FAKE_GHP
                    + "' https://api.github.com/user",
                    "description": "inspect the authenticated user",
                })
                return {"final_response": "done", "messages": []}

        gateway_runner = MagicMock()
        gateway_runner.config = SimpleNamespace(streaming=None)
        gateway_runner._provider_routing = {}
        gateway_runner._agent_cache_lock = None
        gateway_runner._agent_cache = {}
        gateway_runner._session_db = None
        gateway_runner._prefill_messages = None
        gateway_runner._pending_model_notes = {}
        gateway_runner._pending_skills_reload_notes = {}
        gateway_runner.session_store._entries = {}
        gateway_runner._get_system_prompt_for_channel.return_value = None
        gateway_runner._resolve_session_agent_runtime.return_value = ("test-model", {})
        gateway_runner._resolve_session_reasoning_config.return_value = None
        gateway_runner._resolve_session_service_tier.return_value = None
        gateway_runner._resolve_turn_agent_config.return_value = {
            "model": "test-model",
            "runtime": {},
        }
        gateway_runner._agent_config_signature.return_value = ("test-signature",)
        gateway_runner._extract_cache_busting_config.return_value = {}
        gateway_runner._refresh_fallback_model.return_value = None
        gateway_runner._consume_pending_native_image_paths.return_value = []
        gateway_runner._consume_pending_turn_sidecar_notes.return_value = []
        gateway_runner._is_telegram_topic_lane.return_value = False
        gateway_runner._is_discord_auto_thread_lane.return_value = False
        gateway_runner._is_relay_discord_channel_lane.return_value = False

        adapter = _ApprovalAdapter()
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="approval-chat",
            user_id="approval-user",
        )
        ctx = TurnContext(
            source=source,
            message="inspect my account",
            history=[],
            session_id="approval-redaction-session",
            session_key="approval-redaction-session",
            user_config={},
            AIAgent=_ApprovalAgent,
            resolve_display_setting=lambda *_args: False,
            _run_still_current=lambda: True,
            _status_adapter=adapter,
            _status_chat_id="approval-chat",
            _status_thread_metadata={},
            _hooks_ref=SimpleNamespace(loaded_hooks=False),
        )

        def _run_scheduled(coro, *_args, **_kwargs):
            import asyncio

            future = Future()
            future.set_result(asyncio.run(coro))
            return future

        with patch("gateway.run.safe_schedule_threadsafe", side_effect=_run_scheduled):
            result = TurnRunner(gateway_runner, ctx).run_sync()

        assert result["final_response"] == "done"
        assert _FAKE_GHP not in sent["command"]
        assert "curl" in sent["command"]
        assert "github.com" in sent["command"]

    def test_runs_api_path_redacts_pending_approval(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        run_id = "run-redaction"
        entry = approval_mod._ApprovalEntry({
            "request_id": "approval-redaction",
            "command": "curl -H 'Authorization: token " + _FAKE_GHP
            + "' https://api.github.com/user",
            "description": "inspect the authenticated user",
        })
        adapter._run_approval_sessions[run_id] = run_id
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [entry]

        try:
            pending = adapter._pending_run_approval(run_id)
        finally:
            with approval_mod._lock:
                approval_mod._gateway_queues.pop(run_id, None)
            adapter._run_approval_sessions.pop(run_id, None)

        assert pending is not None
        assert pending["request_id"] == "approval-redaction"
        assert _FAKE_GHP not in pending["command"]
        assert "curl" in pending["command"]
        assert "github.com" in pending["command"]


class TestApprovalTextFallbackContract:
    def test_smart_deny_only_advertises_one_operation(self):
        from gateway.run import _format_exec_approval_fallback

        text = _format_exec_approval_fallback(
            "rm -rf /", "dangerous deletion", "/",
            allow_permanent=False, smart_denied=True,
        )
        assert "owner override" in text.lower()
        assert "one operation" in text.lower()
        assert "`/approve`" in text
        assert "approve session" not in text
        assert "approve always" not in text

