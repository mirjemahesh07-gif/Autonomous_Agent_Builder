"""
tool_resolver.py
----------------
ToolResolver: on-demand tool resolution pipeline.

Resolution order
----------------
1. Search the ToolRegistry (exact + fuzzy).
2. Search connected MCP servers for a matching tool.
3. If not found → generate an OpenAPI or MCP schema based on tool name heuristics.
4. Validate the generated schema (structure check + optional test call).
5. Register in the ToolRegistry for future reuse.
6. Return the ToolDefinition.

Constraints
-----------
• NEVER generates raw executable Python code as a tool.
• Only produces structured JSON schemas (OpenAPI path-item or MCP config).
• Validation is always performed before registration.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, TYPE_CHECKING

from tool_registry.registry import ToolRegistry
from utils.logger import get_logger
from utils.schemas import ToolDefinition, ToolType

if TYPE_CHECKING:
    from mcp_manager import MCPManager

log = get_logger("ToolResolver")


# ---------------------------------------------------------------------------
# Built-in schema templates
# ---------------------------------------------------------------------------

def _openapi_schema(name: str, description: str,
                    params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a minimal OpenAPI 3.0 path-item object for a tool."""
    properties = {
        p: {"type": "string", "description": d}
        for p, d in params.items()
    }
    return {
        "openapi": "3.0.0",
        "info": {"title": name, "version": "1.0.0"},
        "paths": {
            f"/{name}": {
                "post": {
                    "operationId": name,
                    "summary": description,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": properties,
                                    "required": list(properties.keys()),
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Success"}
                    },
                }
            }
        },
    }


def _mcp_schema(name: str, description: str,
                params: Dict[str, Any]) -> Dict[str, Any]:
    """Create an MCP tool configuration object."""
    return {
        "name": name,
        "description": description,
        "version": "1.0.0",
        "protocol": "mcp",
        "input_schema": {
            "type": "object",
            "properties": {
                p: {"type": "string", "description": d}
                for p, d in params.items()
            },
            "required": list(params.keys()),
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
            },
        },
        "transport": "stdio",
    }


# ---------------------------------------------------------------------------
# Known-tool catalogue (extends easily — add entries here)
# ---------------------------------------------------------------------------

_TOOL_CATALOGUE: Dict[str, Dict[str, Any]] = {
    "web_search": {
        "type": ToolType.OPENAPI,
        "description": "Search the web and return top results",
        "params": {"query": "The search query", "num_results": "Number of results to return"},
    },
    "calculator": {
        "type": ToolType.MCP,
        "description": "Perform arithmetic calculations",
        "params": {"expression": "Mathematical expression to evaluate"},
    },
    "email_sender": {
        "type": ToolType.MCP,
        "description": "Send an email to a recipient",
        "params": {
            "to": "Recipient email address",
            "subject": "Email subject line",
            "body": "Email body content",
        },
    },
    "file_reader": {
        "type": ToolType.OPENAPI,
        "description": "Read contents of a local or remote file",
        "params": {"path": "File path or URL to read"},
    },
    "summarizer": {
        "type": ToolType.MCP,
        "description": "Summarize a block of text",
        "params": {"text": "Text to summarize", "max_words": "Maximum words in summary"},
    },
    "code_executor": {
        "type": ToolType.MCP,
        "description": "Execute a code snippet in a sandboxed environment",
        "params": {"language": "Programming language", "code": "Code to execute"},
    },
    "data_fetcher": {
        "type": ToolType.OPENAPI,
        "description": "Fetch structured data from a REST API endpoint",
        "params": {"url": "Target API URL", "method": "HTTP method (GET/POST)"},
    },
}


def _infer_catalogue_entry(name: str) -> Dict[str, Any]:
    """
    Generate a generic catalogue entry for any unknown tool name
    by analysing its name.
    """
    name_lower = name.lower()
    if any(kw in name_lower for kw in ("search", "fetch", "get", "query", "lookup")):
        tool_type = ToolType.OPENAPI
    else:
        tool_type = ToolType.MCP

    return {
        "type": tool_type,
        "description": f"Performs the '{name}' operation",
        "params": {"input": f"Primary input for {name}"},
    }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class _SchemaValidator:
    """Validates that a generated schema has the required structure."""

    @staticmethod
    def validate_openapi(schema: Dict) -> bool:
        required_keys = {"openapi", "info", "paths"}
        if not required_keys.issubset(schema):
            log.warning("OpenAPI schema missing keys: %s",
                        required_keys - set(schema))
            return False
        if not schema.get("paths"):
            log.warning("OpenAPI schema has empty 'paths'")
            return False
        return True

    @staticmethod
    def validate_mcp(schema: Dict) -> bool:
        required_keys = {"name", "description", "protocol", "input_schema"}
        if not required_keys.issubset(schema):
            log.warning("MCP schema missing keys: %s",
                        required_keys - set(schema))
            return False
        return True

    @classmethod
    def validate(cls, tool_type: ToolType, schema: Dict) -> bool:
        if tool_type == ToolType.OPENAPI:
            return cls.validate_openapi(schema)
        if tool_type == ToolType.MCP:
            return cls.validate_mcp(schema)
        return True


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

class ToolResolver:
    """Resolves tool names to validated ToolDefinitions."""

    def __init__(self, registry: Optional[ToolRegistry] = None,
                 mcp_manager: Optional["MCPManager"] = None) -> None:
        self._registry = registry or ToolRegistry()
        self._validator = _SchemaValidator()
        self._mcp_manager = mcp_manager

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def resolve(self, tool_name: str,
                context: Optional[Dict[str, Any]] = None) -> ToolDefinition:
        """
        Return a validated ToolDefinition for tool_name.
        Resolution order: registry → MCP servers → local generation.
        """
        log.info("Resolving tool: '%s'", tool_name)

        # 1. Registry lookup
        cached = self._registry.search(tool_name)
        if cached:
            log.info("Tool '%s' resolved from registry.", tool_name)
            return cached

        # 2. MCP server lookup
        if self._mcp_manager:
            mcp_tool = self._mcp_manager.find_tool(tool_name)
            if mcp_tool:
                log.info("Tool '%s' resolved from MCP server '%s' as '%s'.",
                         tool_name, mcp_tool.server_name, mcp_tool.tool_name)
                tool_def = ToolDefinition(
                    name=tool_name,
                    tool_type=ToolType.MCP,
                    schema=mcp_tool.input_schema,
                    mcp_server=mcp_tool.server_name,
                    mcp_tool_name=mcp_tool.tool_name,
                    validated=True,
                    metadata={
                        "description": mcp_tool.description,
                        "source": "mcp_server",
                        "mcp_server": mcp_tool.server_name,
                    },
                )
                # Cache in registry
                self._registry.register(tool_def)
                return tool_def

        # 3. Generate schema locally
        tool_def = self._generate(tool_name)

        # 4. Validate
        is_valid = self._validator.validate(tool_def.tool_type, tool_def.schema)
        tool_def.validated = is_valid

        if not is_valid:
            log.error("Generated schema for '%s' failed validation!", tool_name)
        else:
            log.info("Tool '%s' schema validated successfully.", tool_name)

        # 5. Register (even if invalid — marks it so callers can decide)
        self._registry.register(tool_def)

        return tool_def

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _generate(self, tool_name: str) -> ToolDefinition:
        """Build an OpenAPI or MCP schema for tool_name."""
        entry = _TOOL_CATALOGUE.get(tool_name) or _infer_catalogue_entry(tool_name)
        tool_type: ToolType = entry["type"]
        description: str    = entry["description"]
        params: Dict        = entry["params"]

        if tool_type == ToolType.OPENAPI:
            schema = _openapi_schema(tool_name, description, params)
        else:
            schema = _mcp_schema(tool_name, description, params)

        log.debug("Generated %s schema for '%s':\n%s",
                  tool_type.value, tool_name, json.dumps(schema, indent=2))

        return ToolDefinition(
            name=tool_name,
            tool_type=tool_type,
            schema=schema,
            validated=False,
            metadata={
                "description": description,
                "auto_generated": True,
                "source": "catalogue" if tool_name in _TOOL_CATALOGUE else "inferred",
            },
        )
