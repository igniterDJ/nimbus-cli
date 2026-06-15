"""MCP stdio client for nimbus — optional dep. Import-guarded in nimbus.py."""
import asyncio
import sys
import threading
from typing import Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _MCP_SDK = True
except ImportError:
    _MCP_SDK = False


class McpManager:
    """Run MCP stdio servers in a background asyncio event loop."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, list] = {}
        self._errors: dict[str, str] = {}
        self._stop_events: dict[str, asyncio.Event] = {}

    def _run(self, coro: Any, timeout: float = 30) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def connect_all(self, servers: dict) -> None:
        if not _MCP_SDK:
            return
        for name, cfg in servers.items():
            try:
                self._run(self._connect_server(name, cfg), timeout=30)
            except Exception as exc:
                self._errors[name] = str(exc)
                print(f"[nimbus] warn: MCP server {name!r} failed to start: {exc}", file=sys.stderr)

    async def _connect_server(self, name: str, cfg: dict) -> None:
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=cfg.get("env") or None,
        )
        loop = self._loop
        ready: asyncio.Future = loop.create_future()
        stop: asyncio.Event = asyncio.Event()
        self._stop_events[name] = stop

        async def _run_conn() -> None:
            try:
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools_result = await session.list_tools()
                        self._sessions[name] = session
                        self._tools[name] = tools_result.tools
                        if not ready.done():
                            ready.set_result(True)
                        await stop.wait()
            except Exception as exc:
                if not ready.done():
                    ready.set_exception(exc)

        asyncio.ensure_future(_run_conn(), loop=loop)
        await ready

    def tool_schemas(self) -> list[dict]:
        schemas = []
        for server_name, tools in self._tools.items():
            for tool in tools:
                full_name = f"mcp__{server_name}__{tool.name}"
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": full_name,
                        "description": f"[MCP:{server_name}] {tool.description or ''}",
                        "parameters": tool.inputSchema if tool.inputSchema else {
                            "type": "object", "properties": {}
                        },
                    },
                })
        return schemas

    def call(self, full_name: str, args: dict) -> str:
        parts = full_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            return f"ERROR: invalid MCP tool name: {full_name}"
        server_name, tool_name = parts[1], parts[2]
        session = self._sessions.get(server_name)
        if session is None:
            return f"ERROR: MCP server {server_name!r} not connected"
        try:
            result = self._run(session.call_tool(tool_name, args))
            texts = []
            for content in result.content:
                if hasattr(content, "text"):
                    texts.append(content.text)
                else:
                    texts.append(f"[non-text: {type(content).__name__}]")
            return "\n".join(texts) if texts else "(empty result)"
        except Exception as exc:
            return f"ERROR: MCP call failed: {exc}"

    def status(self) -> str:
        lines = []
        for name, tools in self._tools.items():
            tool_names = ", ".join(t.name for t in tools[:5])
            suffix = f"... +{len(tools)-5} more" if len(tools) > 5 else ""
            lines.append(f"  ✓ {name}: {len(tools)} tool(s) — {tool_names}{suffix}")
        for name, error in self._errors.items():
            lines.append(f"  ✗ {name}: {error}")
        if not lines:
            lines.append("  (no MCP servers configured or connected)")
        return "\n".join(lines)

    def close(self) -> None:
        for stop in self._stop_events.values():
            self._loop.call_soon_threadsafe(stop.set)
        self._loop.call_soon_threadsafe(self._loop.stop)
