"""
execution_engine.py
-------------------
ExecutionEngine: drives a single Agent through its task lifecycle.

Responsibilities
----------------
• Decide whether a tool call is needed (based on task.required_tools).
• Execute the tool via MCP if available, otherwise fall back to simulation.
• Build structured AgentOutput with result, status, and used_tools.
• Retry on transient failures (exponential back-off).
• Never generates raw executable Python code.

The engine is injected into each Agent by the Orchestrator before run().
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import config
from utils.logger import get_logger
from utils.schemas import AgentOutput, TaskStatus, ToolDefinition

if TYPE_CHECKING:
    from mcp_manager import MCPManager

log = get_logger("ExecutionEngine")


# ---------------------------------------------------------------------------
# Simulated tool dispatcher (fallback when MCP is not available)
# ---------------------------------------------------------------------------

def _simulate_tool_call(tool: ToolDefinition,
                         inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a plausible fake response for a tool call.
    Used as fallback when the tool has no MCP server attached.
    """
    name = tool.name
    if name == "web_search":
        query = inputs.get("context", "general topic")
        return {
            "results": [
                {"title": f"Result 1 for '{query}'", "snippet": "First result …"},
                {"title": f"Result 2 for '{query}'", "snippet": "Second result …"},
            ]
        }
    if name == "calculator":
        expression = inputs.get("expression", "0")
        try:
            # Safe eval restricted to arithmetic
            result = eval(  # noqa: S307
                expression,
                {"__builtins__": {}},
                {"abs": abs, "round": round, "pow": pow},
            )
            return {"result": result}
        except Exception:
            return {"result": None, "error": "Could not evaluate expression"}
    if name == "email_sender":
        return {"sent": True, "message_id": "sim-12345"}
    if name == "summarizer":
        text = inputs.get("text", "")
        return {"summary": text[:120] + ("…" if len(text) > 120 else "")}
    if name == "file_reader":
        return {"content": f"[Simulated file content for path: {inputs.get('path', '?')}]"}

    # Generic fallback
    return {"output": f"[Simulated output from tool '{name}']"}


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """Executes a single agent with retry logic and structured output."""

    def __init__(self, mcp_manager: Optional["MCPManager"] = None) -> None:
        self._max_retries    = config.MAX_RETRIES
        self._backoff        = config.RETRY_BACKOFF_SECONDS
        self._mcp            = mcp_manager

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    async def execute(self, agent, inputs: Dict[str, Any]) -> AgentOutput:
        """
        Run the agent's task with retries.

        Parameters
        ----------
        agent  : Agent   (imported lazily to avoid circular import)
        inputs : dict    upstream task outputs passed as context
        """
        task      = agent.task
        tools     = agent.tools
        task_id   = task.id

        log.info("▶  Executing task '%s': %s", task_id, task.description)

        for attempt in range(1, self._max_retries + 2):
            try:
                result, used_tools = await self._run_once(task, tools, inputs)
                output = AgentOutput(
                    task_id=task_id,
                    result=result,
                    status=TaskStatus.COMPLETED,
                    used_tools=used_tools,
                )
                log.info("✔  Task '%s' completed (attempt %d).", task_id, attempt)
                return output

            except Exception as exc:
                if attempt > self._max_retries:
                    log.error("✘  Task '%s' failed after %d attempts: %s",
                              task_id, attempt, exc)
                    return AgentOutput(
                        task_id=task_id,
                        result=None,
                        status=TaskStatus.FAILED,
                        error=str(exc),
                    )
                wait = self._backoff * (2 ** (attempt - 1))
                log.warning("⚠  Task '%s' attempt %d failed (%s). "
                            "Retrying in %.1fs …", task_id, attempt, exc, wait)
                await asyncio.sleep(wait)

        # Should never reach here
        return AgentOutput(task_id=task_id, result=None,
                           status=TaskStatus.FAILED, error="Unknown error")

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    async def _run_once(self, task, tools: List[ToolDefinition],
                        inputs: Dict[str, Any]):
        """Single execution attempt — returns (result_dict, used_tool_names)."""
        used_tools: List[str] = []
        tool_results: Dict[str, Any] = {}

        # --- Invoke tools if required ---------------------------------- #
        for tool in tools:
            log.debug("  Calling tool '%s' for task '%s' …",
                      tool.name, task.id)

            if tool.mcp_server and self._mcp:
                # ── Real MCP call ──────────────────────────────────────
                tool_result = await self._execute_mcp_tool(tool, inputs)
            else:
                # ── Simulated fallback ─────────────────────────────────
                tool_result = _simulate_tool_call(tool, inputs)

            tool_results[tool.name] = tool_result
            used_tools.append(tool.name)
            log.debug("  Tool '%s' result: %s", tool.name,
                      json.dumps(tool_result, default=str)[:200])

        # --- Core task logic ------------------------------------------ #
        result = self._reason(task, inputs, tool_results)

        return result, used_tools

    async def _execute_mcp_tool(self, tool: ToolDefinition,
                                inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Route tool call through the MCPManager."""
        assert self._mcp is not None
        assert tool.mcp_server is not None
        assert tool.mcp_tool_name is not None

        # Build arguments from the agent's inputs.
        # The inputs dict may contain upstream task outputs keyed by task_id.
        # We flatten and pass relevant data to the MCP tool.
        arguments = self._build_mcp_arguments(tool, inputs)

        try:
            result = await self._mcp.call_tool(
                tool.mcp_server,
                tool.mcp_tool_name,
                arguments,
            )
            return result
        except Exception as exc:
            log.warning("MCP tool '%s' on server '%s' failed: %s. "
                        "Falling back to simulation.",
                        tool.mcp_tool_name, tool.mcp_server, exc)
            return _simulate_tool_call(tool, inputs)

    @staticmethod
    def _build_mcp_arguments(tool: ToolDefinition,
                             inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract arguments for an MCP tool from the agent's inputs.

        Strategy:
        - If inputs contain a key matching the tool name, use that.
        - If the tool schema has 'properties', try to map input keys.
        - Otherwise pass a serialised context string.
        """
        # Check if there's direct tool input
        if tool.name in inputs:
            val = inputs[tool.name]
            if isinstance(val, dict):
                return val

        # Flatten upstream outputs into a context string for generic tools
        context = json.dumps(inputs, default=str)[:2000]

        # If the tool's input_schema specifies properties, try to fill them
        schema = tool.schema
        if isinstance(schema, dict):
            properties = schema.get("properties", {})
            if not properties:
                # Nested under input_schema for MCP-style schemas
                properties = schema.get("input_schema", {}).get("properties", {})

            if properties:
                args: Dict[str, Any] = {}
                for param_name in properties:
                    if param_name in inputs:
                        args[param_name] = inputs[param_name]
                if args:
                    return args

        # Fallback: pass context as a generic input
        return {"input": context}

    @staticmethod
    def _reason(task, inputs: Dict[str, Any],
                tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate task reasoning / LLM call.
        In production, replace with a real LLM call using litellm.
        """
        # Build a structured summary that downstream tasks can use
        return {
            "task_id":      task.id,
            "description":  task.description,
            "status":       "completed",
            "inputs_used":  list(inputs.keys()),
            "tool_outputs": tool_results,
            "output":       (
                f"[Result of: {task.description}] "
                + (
                    f"Using tool outputs: {list(tool_results.keys())}"
                    if tool_results else "No external tools needed."
                )
            ),
        }
