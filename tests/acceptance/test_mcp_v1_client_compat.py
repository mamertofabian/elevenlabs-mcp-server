"""An actual SDK-v1 client negotiates with the current SDK-v2 stdio server."""

from __future__ import annotations

import os
import subprocess
import sys


def test_legacy_mcp_client_can_discover_plan_and_read_history(tmp_path):
    environment = tmp_path / "client"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    python = environment / "bin/python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "mcp==1.1.2"],
        check=True,
        capture_output=True,
        timeout=180,
    )
    code = r"""
import asyncio,json,os,sys
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl
async def run():
    params=StdioServerParameters(command=os.environ['REVIVAL_SERVER_PYTHON'],args=['-m','elevenlabs_mcp.server'],env=dict(os.environ))
    async with stdio_client(params) as (read,write),ClientSession(read,write) as session:
        init=await session.initialize()
        assert init.serverInfo.name=='elevenlabs-server'
        names={t.name for t in (await session.list_tools()).tools}
        assert {'generate_audio_simple','plan_voiceover','resume_voiceover','get_artifact'}<=names
        result=await session.call_tool('plan_voiceover',{'script':{'script_version':'1','cast':{'a':{'voice_id':'voice'}},'scenes':[{'id':'s','parts':[{'id':'p','actor':'a','text':'Hello.'}]}]},'options':{'engine':'tts','model_id':'eleven_multilingual_v2'}})
        assert json.loads(result.content[0].text)['ok'] is True
        history=await session.read_resource(AnyUrl('voiceover://history'))
        assert history.contents[0].mimeType=='text/plain'
        assert json.loads(history.contents[0].text)==[]
asyncio.run(run())
"""
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=tmp_path,
        env={
            **os.environ,
            "REVIVAL_SERVER_PYTHON": sys.executable,
            "ELEVENLABS_API_KEY": "",
            "ELEVENLABS_DATABASE_PATH": str(tmp_path / "state.db"),
            "ELEVENLABS_OUTPUT_DIR": str(tmp_path / "out"),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
