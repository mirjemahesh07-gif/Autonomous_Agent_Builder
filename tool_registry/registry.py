"""
tool_registry/registry.py
--------------------------
Persistent, file-system-backed Tool Registry.

Directory layout
----------------
tool_registry/
    openapi/     → <tool_name>.json   (OpenAPI path-item schemas)
    mcp/         → <tool_name>.json   (MCP tool configs)
    metadata/    → <tool_name>.json   (validation status + usage stats)

Public API
----------
registry = ToolRegistry()
registry.register(tool_def)
registry.search("web_search")          -> ToolDefinition | None
registry.retrieve("web_search")        -> ToolDefinition
registry.list_all()                    -> List[ToolDefinition]
"""

from __future__ import annotations

import json
import os
from difflib import get_close_matches
from typing import Dict, List, Optional

import config
from utils.logger import get_logger
from utils.schemas import ToolDefinition, ToolType

log = get_logger("ToolRegistry")


class ToolRegistry:
    """Directory-backed store for validated tool definitions."""

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        self._ensure_dirs()
        log.debug("ToolRegistry initialised at %s", config.TOOL_REGISTRY_ROOT)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def register(self, tool: ToolDefinition) -> None:
        """Persist a tool definition to disk (overwrites if already exists)."""
        schema_dir = self._schema_dir(tool.tool_type)
        schema_path = os.path.join(schema_dir, f"{tool.name}.json")
        meta_path   = os.path.join(config.TOOL_REGISTRY_METADATA_DIR,
                                   f"{tool.name}.json")

        # Write schema
        with open(schema_path, "w", encoding="utf-8") as fh:
            json.dump(tool.schema, fh, indent=2)

        # Write metadata
        meta: Dict = {
            "name":       tool.name,
            "tool_type":  tool.tool_type.value,
            "validated":  tool.validated,
            "schema_path": schema_path,
            **tool.metadata,
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        log.info("Registered tool '%s' (%s)", tool.name, tool.tool_type.value)

    def search(self, name: str) -> Optional[ToolDefinition]:
        """
        Exact-match first, then fuzzy fallback.
        Returns None if nothing suitable is found.
        """
        # 1. Exact match
        tool = self._load_by_name(name)
        if tool:
            log.debug("Tool '%s' found in registry (exact match)", name)
            return tool

        # 2. Fuzzy match
        all_names = self._list_registered_names()
        matches = get_close_matches(name, all_names, n=1, cutoff=0.6)
        if matches:
            log.debug("Tool '%s' not found; fuzzy-matched to '%s'",
                      name, matches[0])
            return self._load_by_name(matches[0])

        log.debug("Tool '%s' not found in registry", name)
        return None

    def retrieve(self, name: str) -> ToolDefinition:
        """Retrieve a tool by exact name; raises KeyError if missing."""
        tool = self._load_by_name(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry")
        return tool

    def list_all(self) -> List[ToolDefinition]:
        """Return every registered tool."""
        tools: List[ToolDefinition] = []
        for name in self._list_registered_names():
            t = self._load_by_name(name)
            if t:
                tools.append(t)
        return tools

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_dirs(self) -> None:
        for d in (
            config.TOOL_REGISTRY_ROOT,
            config.TOOL_REGISTRY_OPENAPI_DIR,
            config.TOOL_REGISTRY_MCP_DIR,
            config.TOOL_REGISTRY_METADATA_DIR,
        ):
            os.makedirs(d, exist_ok=True)

    def _schema_dir(self, tool_type: ToolType) -> str:
        return {
            ToolType.OPENAPI: config.TOOL_REGISTRY_OPENAPI_DIR,
            ToolType.MCP:     config.TOOL_REGISTRY_MCP_DIR,
            ToolType.BUILTIN: config.TOOL_REGISTRY_OPENAPI_DIR,
        }[tool_type]

    def _list_registered_names(self) -> List[str]:
        meta_dir = config.TOOL_REGISTRY_METADATA_DIR
        if not os.path.isdir(meta_dir):
            return []
        return [
            f[:-5]                       # strip .json
            for f in os.listdir(meta_dir)
            if f.endswith(".json")
        ]

    def _load_by_name(self, name: str) -> Optional[ToolDefinition]:
        meta_path = os.path.join(config.TOOL_REGISTRY_METADATA_DIR,
                                 f"{name}.json")
        if not os.path.exists(meta_path):
            return None

        with open(meta_path, encoding="utf-8") as fh:
            meta: Dict = json.load(fh)

        tool_type = ToolType(meta.get("tool_type", "openapi"))
        schema_path: str = meta.get("schema_path", "")
        schema: Dict = {}
        if schema_path and os.path.exists(schema_path):
            with open(schema_path, encoding="utf-8") as fh:
                schema = json.load(fh)

        return ToolDefinition(
            name=meta["name"],
            tool_type=tool_type,
            schema=schema,
            validated=meta.get("validated", False),
            metadata={k: v for k, v in meta.items()
                      if k not in ("name", "tool_type", "validated",
                                   "schema_path")},
        )
