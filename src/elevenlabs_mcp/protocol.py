"""Public SDK-v2 callbacks with registration compatible with legacy handlers."""

from __future__ import annotations

import logging
from typing import Any

from mcp import types
from mcp.server import Server


class ProtocolServer:
    def __init__(self, name: str):
        self.name = name
        self.handlers: dict[str, Any] = {}
        self.revival: Any = None

    def _register(self, key):
        def decorate(handler):
            self.handlers[key] = handler
            return handler

        return decorate

    def list_tools(self):
        return self._register("tools")

    def call_tool(self):
        return self._register("call")

    def list_resource_templates(self):
        return self._register("templates")

    def read_resource(self):
        return self._register("read")

    def progress_notification(self):
        return self._register("progress")

    def _build(self):
        async def resources(_ctx, _params):
            return types.ListResourcesResult(
                resources=[
                    types.Resource(
                        uri="voiceover://history",
                        name="Voiceover history",
                        mime_type="text/plain",
                    ),
                    types.Resource(
                        uri="voiceover://voices",
                        name="Voice metadata",
                        mime_type="text/plain",
                    ),
                ]
            )

        async def tools(_ctx, _params):
            catalog = await self.handlers["tools"]()
            if self.revival is not None:
                catalog += self.revival.tools()
            return types.ListToolsResult(tools=catalog)

        async def call(_ctx, params):
            if self.revival is not None and self.revival.handles(params.name):
                return await self.revival.call(params.name, params.arguments or {})
            content = await self.handlers["call"](params.name, params.arguments or {})
            return types.CallToolResult(content=content)

        async def templates(_ctx, _params):
            values = await self.handlers["templates"]()
            if self.revival is not None:
                values += [
                    types.ResourceTemplate(
                        uri_template="voiceover://artifacts/{artifact_id}",
                        name="Production artifact",
                        mime_type="application/json",
                    )
                ]
            return types.ListResourceTemplatesResult(resource_templates=values)

        async def read(_ctx, params):
            if self.revival is not None and str(params.uri).startswith(
                "voiceover://artifacts/"
            ):
                return await self.revival.read_resource(params.uri)
            if self.revival is not None and str(params.uri) == "voiceover://voices":
                result = await self.revival.call("list_voices", {})
                return types.ReadResourceResult(
                    contents=[
                        types.TextResourceContents(
                            uri=params.uri,
                            mime_type="text/plain",
                            text=result.content[0].text,
                        )
                    ]
                )
            if self.revival is not None and str(params.uri).startswith(
                "voiceover://history"
            ):
                job_id = (
                    str(params.uri).removeprefix("voiceover://history").strip("/")
                    or None
                )
                import json

                text = json.dumps(await self.revival._history(job_id))
            else:
                text = await self.handlers["read"](params.uri)
            return types.ReadResourceResult(
                contents=[
                    types.TextResourceContents(
                        uri=params.uri, mime_type="text/plain", text=text
                    )
                ]
            )

        async def progress(_ctx, params):
            handler = self.handlers.get("progress")
            if handler:
                try:
                    await handler(params.progress_token, params.progress, params.total)
                except Exception:  # noqa: BLE001 - sanitize failures at the IO/protocol boundary
                    logging.getLogger(__name__).warning("Progress handler failed")

        server = Server(
            self.name,
            on_list_resources=resources,
            on_list_tools=tools,
            on_call_tool=call,
            on_list_resource_templates=templates,
            on_read_resource=read,
            on_progress=progress if "progress" in self.handlers else None,
        )
        server.middleware = []  # No telemetry is installed or exported.
        return server

    def create_initialization_options(self, **kwargs):
        return self._build().create_initialization_options(**kwargs)

    def get_capabilities(self, **kwargs):
        return self._build().get_capabilities(**kwargs)

    async def run(
        self, read_stream, write_stream, initialization_options, raise_exceptions=False
    ):
        await self._build().run(
            read_stream,
            write_stream,
            initialization_options,
            raise_exceptions=raise_exceptions,
        )
