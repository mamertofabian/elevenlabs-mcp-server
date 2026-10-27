import json
import os
import subprocess
import sys

EXPECTED_TOOLS = {
    "generate_audio_simple",
    "generate_audio_script",
    "delete_job",
    "get_audio_file",
    "list_voices",
    "get_voiceover_history",
}


def _rpc(message):
    return (json.dumps(message) + "\n").encode()


def test_server_answers_initialize_and_lists_tools_over_stdio(tmp_path):
    env = dict(os.environ, ELEVENLABS_API_KEY="test-key")
    proc = subprocess.Popen(
        [sys.executable, "-c", "from elevenlabs_mcp.server import main; main()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=env,
    )
    requests = (
        _rpc(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "smoke-test", "version": "1"},
                },
            }
        )
        + _rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
        + _rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    )
    try:
        stdout, stderr = proc.communicate(requests, timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    responses = {}
    for line in stdout.decode().splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in message:
            responses[message["id"]] = message

    assert 1 in responses, f"no initialize response; stderr:\n{stderr.decode()[-2000:]}"
    assert responses[1]["result"]["serverInfo"]["name"] == "elevenlabs-server"
    assert 2 in responses, f"no tools/list response; stderr:\n{stderr.decode()[-2000:]}"
    tool_names = {tool["name"] for tool in responses[2]["result"]["tools"]}
    assert tool_names == EXPECTED_TOOLS
