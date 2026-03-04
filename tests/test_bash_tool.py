import asyncio
from pathlib import Path
import shutil

import pytest

from pi_coding_agent.tools.bash import create_bash_tool


def run_execute(cwd: Path, arguments: dict) -> list[dict]:
    tool = create_bash_tool(str(cwd))
    execute = tool["execute"]
    return asyncio.run(execute("test-call-id", arguments))


def test_bash_executes_command_and_returns_output(tmp_path: Path) -> None:
    result = run_execute(tmp_path, {"command": "echo hello"})

    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert "hello" in result[0]["text"]


def test_bash_executes_command_with_no_output(tmp_path: Path) -> None:
    result = run_execute(tmp_path, {"command": "true"})

    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "(no output)"


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl not available")
def test_bash_executes_curl_command(tmp_path: Path) -> None:
    result = run_execute(tmp_path, {"command": "curl --version"})

    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert "curl" in result[0]["text"].lower()


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl not available")
def test_bash_executes_curl_http_and_captures_output(tmp_path: Path) -> None:
    command = "curl https://cards.iqiyi.com/views_plt/3.0/echo?action=getA&value=2444700691711401"

    try:
        # 如果 curl 返回 0，我们应该拿到正常输出
        result = run_execute(tmp_path, {"command": command})
    except RuntimeError as exc:
        # 如果 curl 返回非 0，我们应该从异常消息中看到 curl 的 stderr
        msg = str(exc).lower()
        assert "curl" in msg
        return

    # 成功返回时，应该有文本输出
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"]


def test_bash_raises_on_non_zero_exit_code(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        run_execute(tmp_path, {"command": "false"})

    msg = str(exc_info.value)
    assert "Command exited with code" in msg


def test_bash_raises_when_cwd_does_not_exist(tmp_path: Path) -> None:
    non_existing = tmp_path / "does-not-exist"

    tool = create_bash_tool(str(non_existing))
    execute = tool["execute"]

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(execute("test-call-id", {"command": "echo hello"}))

    assert "Working directory does not exist" in str(exc_info.value)

