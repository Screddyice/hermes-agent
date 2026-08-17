"""Regression tests for hermetic guards around local desktop side effects."""

from __future__ import annotations

import json
import subprocess
import sys
import webbrowser


def test_child_processes_resolve_the_no_op_browser():
    """A monkeypatch stops at the process boundary; ``$BROWSER`` does not.

    The suite runs one pytest subprocess per test file and shells out to the
    ``hermes`` CLI, so the in-process fixture below cannot be the only guard.
    Resolve the browser inside a *child* interpreter and assert it is the no-op
    rather than the platform default — on macOS that default shells out to
    ``osascript`` and opens a real OAuth consent page.
    """
    probe = (
        "import json, webbrowser;"
        "c = webbrowser.get();"
        "print(json.dumps({'cls': type(c).__name__,"
        " 'name': getattr(c, 'name', None)}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = json.loads(result.stdout)

    assert resolved["cls"] == "GenericBrowser"
    assert resolved["name"] == "true"


def test_child_processes_do_not_see_the_real_qwen_credentials():
    """``~/.qwen/oauth_creds.json`` sits outside the ``HERMES_HOME`` sandbox.

    ``_qwen_cli_auth_path()`` resolves it off ``Path.home()``, and this suite
    deliberately does not redirect HOME, so without an environment-level
    override a developer's real Qwen token reaches
    ``load_pool("qwen-oauth")`` mid-test. ``resolve_runtime_provider``
    consults that pool for any ``auto`` request, which both leaks the
    credential and makes provider-resolution tests depend on whether the
    machine happens to be logged into Qwen. Probe a child interpreter, since
    that is exactly what an in-process patch cannot cover.
    """
    probe = (
        "import json, pathlib;"
        "from hermes_cli.auth import _qwen_cli_auth_path;"
        "p = _qwen_cli_auth_path();"
        "real = pathlib.Path.home() / '.qwen';"
        "print(json.dumps({'under_real_home': str(p).startswith(str(real)),"
        " 'exists': p.exists()}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = json.loads(result.stdout)

    assert resolved["under_real_home"] is False
    assert resolved["exists"] is False


def test_webbrowser_open_calls_are_neutralized(monkeypatch):
    """OAuth/browser tests should never reach the real browser registry."""

    def _real_browser_lookup_reached(*_args, **_kwargs):
        raise AssertionError("test reached the real webbrowser registry")

    monkeypatch.setattr(webbrowser, "get", _real_browser_lookup_reached)

    url = "https://provider.example.invalid/oauth/authorize"

    assert webbrowser.open(url) is True
    assert webbrowser.open_new(url) is True
    assert webbrowser.open_new_tab(url) is True


def test_webbrowser_get_controller_is_neutralized(_neutralize_webbrowser):
    """Direct controller access should still stay inside the test recorder."""
    url = "https://provider.example.invalid/oauth/authorize"

    controller = webbrowser.get("hermes-test-browser")

    assert controller.open(url) is True
    assert controller.open_new(url) is True
    assert controller.open_new_tab(url) is True
    assert _neutralize_webbrowser == [url, url, url]


def _isolate_anthropic_credentials(monkeypatch, tmp_path):
    from agent import anthropic_adapter as aa

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(aa.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(aa.platform, "system", lambda: "Darwin")

    def _real_keychain_reached(*_args, **_kwargs):
        raise AssertionError("test reached the real macOS Keychain command")

    monkeypatch.setattr(aa.subprocess, "run", _real_keychain_reached)
    return aa


def test_claude_code_credential_read_does_not_touch_macos_keychain(
    monkeypatch, tmp_path
):
    """The real credential reader should be safe under the suite guard."""
    aa = _isolate_anthropic_credentials(monkeypatch, tmp_path)

    assert aa.read_claude_code_credentials() is None


def test_anthropic_token_resolution_does_not_touch_macos_keychain(
    monkeypatch, tmp_path
):
    """Token resolution should be safe under the same suite guard."""
    aa = _isolate_anthropic_credentials(monkeypatch, tmp_path)

    assert aa.resolve_anthropic_token() is None
