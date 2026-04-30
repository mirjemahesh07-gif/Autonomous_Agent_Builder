"""
utils/schemas.py
----------------
Shared data-contract models used across every module.
Using dataclasses (stdlib) so there are no mandatory Pydantic imports,
but Pydantic validation is added as an optional enhancement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"


class ToolType(str, Enum):
    OPENAPI = "openapi"
    MCP     = "mcp"
    BUILTIN = "builtin"


# ---------------------------------------------------------------------------
# Planner output
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single unit of work produced by the Planner."""
    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "depends_on": self.depends_on,
            "required_tools": self.required_tools,
        }


@dataclass
class PlannerOutput:
    """Full output from the Planner: tasks + dependency edges."""
    tasks: List[Task]
    dependencies: List[List[str]]   # [[from_id, to_id], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks": [t.to_dict() for t in self.tasks],
            "dependencies": self.dependencies,
        }


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """A resolved, validated tool ready for agent use."""
    name: str
    tool_type: ToolType
    schema: Dict[str, Any]           # OpenAPI path object OR MCP config
    mcp_server: Optional[str] = None      # MCP server name from config
    mcp_tool_name: Optional[str] = None   # tool name on the MCP server
    validated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "tool_type": self.tool_type.value,
            "schema": self.schema,
            "validated": self.validated,
            "metadata": self.metadata,
        }
        if self.mcp_server:
            d["mcp_server"] = self.mcp_server
            d["mcp_tool_name"] = self.mcp_tool_name
        return d


# ---------------------------------------------------------------------------
# Agent models
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration handed to an Agent at creation time."""
    task: Task
    tools: List[ToolDefinition] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Structured result from a single agent execution."""
    task_id: str
    result: Any
    status: TaskStatus
    used_tools: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "result": self.result,
            "status": self.status.value,
            "used_tools": self.used_tools,
            "error": self.error,
        }
