"""
mcp_manager.py
--------------
MCPManager: connects to MCP servers, discovers tools, and executes tool calls.

Responsibilities
----------------
• Read server definitions from mcp_servers.json.
• Launch each MCP server as a subprocess (stdio transport).
• Initialise a ClientSession per server and discover available tools.
• Provide a unified ``call_tool(server, tool, args)`` interface.
• Gracefully shut down all connections on exit.

All public methods are async because the MCP Python SDK is fully async.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config
from utils.logger import get_logger

log = get_logger("MCPManager")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MCPToolInfo:
    """Lightweight descriptor for a tool discovered on an MCP server."""
    server_name: str
    tool_name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class _ServerConnection:
    """Internal state for one MCP server connection."""
    name: str
    session: ClientSession
    tools: List[MCPToolInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class MCPManager:
    """Manages lifecycle and tool calls for multiple MCP servers."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config_path = config_path or config.MCP_SERVERS_CONFIG
        self._exit_stack = AsyncExitStack()
        self._connections: Dict[str, _ServerConnection] = {}
        self._all_tools: List[MCPToolInfo] = []

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def connect_all(self) -> None:
        """Connect to every server listed in the config file."""
        server_defs = self._load_config()

        for name, defn in server_defs.items():
            try:
                await self._connect_one(name, defn)
            except Exception as exc:
                log.warning("⚠  Could not connect to MCP server '%s': %s",
                            name, exc)

        log.info("MCPManager: %d server(s) connected, %d tool(s) discovered.",
                 len(self._connections), len(self._all_tools))

    async def disconnect_all(self) -> None:
        """Shut down all server connections gracefully."""
        log.info("MCPManager: disconnecting all servers …")
        await self._exit_stack.aclose()
        self._connections.clear()
        self._all_tools.clear()
        log.info("MCPManager: all connections closed.")

    # ------------------------------------------------------------------ #
    #  Tool discovery                                                      #
    # ------------------------------------------------------------------ #

    def get_all_tools(self) -> List[MCPToolInfo]:
        """Return a flat list of every tool across all connected servers."""
        return list(self._all_tools)

    def find_tool(self, tool_name: str) -> Optional[MCPToolInfo]:
        """
        Find a tool by name across all connected servers.
        Returns the first match, or None.
        """
        for t in self._all_tools:
            if t.tool_name == tool_name:
                return t
        # Fuzzy: try substring match
        tool_lower = tool_name.lower().replace("_", "").replace("-", "")
        for t in self._all_tools:
            candidate = t.tool_name.lower().replace("_", "").replace("-", "")
            if tool_lower in candidate or candidate in tool_lower:
                log.debug("Fuzzy-matched tool '%s' → '%s' on server '%s'.",
                          tool_name, t.tool_name, t.server_name)
                return t
        return None

    # ------------------------------------------------------------------ #
    #  Tool execution                                                      #
    # ------------------------------------------------------------------ #

    async def call_tool(self, server_name: str, tool_name: str,
                        arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on a specific MCP server.

        Returns the parsed content from the MCP CallToolResult.
        """
        conn = self._connections.get(server_name)
        if conn is None:
            raise RuntimeError(
                f"MCP server '{server_name}' is not connected."
            )

        log.info("  MCP call: server='%s', tool='%s', args=%s",
                 server_name, tool_name,
                 json.dumps(arguments, default=str)[:200])

        result = await conn.session.call_tool(tool_name, arguments=arguments)

        # Parse the CallToolResult content blocks into a dict
        return self._parse_result(result)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _load_config(self) -> Dict[str, Dict[str, Any]]:
        """Load and return the servers dict from mcp_servers.json."""
        if not os.path.exists(self._config_path):
            log.warning("MCP config file not found: %s", self._config_path)
            return {}

        with open(self._config_path, encoding="utf-8") as fh:
            data = json.load(fh)

        servers = data.get("servers", {})
        log.debug("Loaded %d MCP server definition(s) from %s",
                  len(servers), self._config_path)
        return servers

    async def _connect_one(self, name: str, defn: Dict[str, Any]) -> None:
        """Launch one MCP server, initialise session, discover tools."""
        command = defn["command"]
        args = defn.get("args", [])
        env_vars = defn.get("env", None)

        log.info("Connecting to MCP server '%s' → %s %s",
                 name, command, " ".join(args))

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env_vars,
        )

        # stdio_client returns (read_stream, write_stream) via context manager
        transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = transport

        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        # Discover tools
        tools_result = await session.list_tools()
        tools: List[MCPToolInfo] = []
        for t in tools_result.tools:
            info = MCPToolInfo(
                server_name=name,
                tool_name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema if t.inputSchema else {},
            )
            tools.append(info)
            log.debug("  Discovered tool: '%s' on server '%s'",
                      t.name, name)

        conn = _ServerConnection(name=name, session=session, tools=tools)
        self._connections[name] = conn
        self._all_tools.extend(tools)

        log.info("✔  MCP server '%s': %d tool(s) discovered.", name, len(tools))

    @staticmethod
    def _parse_result(result) -> Dict[str, Any]:
        """
        Convert MCP CallToolResult into a plain dict.
        Handles TextContent, ImageContent, EmbeddedResource blocks.
        """
        from mcp import types as mcp_types

        parsed: Dict[str, Any] = {}
        texts: List[str] = []

        if hasattr(result, "structuredContent") and result.structuredContent:
            parsed["structured"] = result.structuredContent

        for block in (result.content or []):
            if isinstance(block, mcp_types.TextContent):
                texts.append(block.text)
            elif isinstance(block, mcp_types.ImageContent):
                parsed.setdefault("images", []).append({
                    "mimeType": block.mimeType,
                    "data_length": len(block.data) if block.data else 0,
                })
            elif isinstance(block, mcp_types.EmbeddedResource):
                resource = block.resource
                if hasattr(resource, "text"):
                    texts.append(resource.text)

        if texts:
            parsed["text"] = "\n".join(texts)

        if result.isError:
            parsed["error"] = True

        return parsed or {"text": "(empty MCP response)"}
