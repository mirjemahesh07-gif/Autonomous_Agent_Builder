"""
planner.py
----------
Planner: converts a free-text user goal into a structured task list + DAG edges.

Two modes
---------
1. LLM mode  (USE_LLM_PLANNER=true)
   Calls an LLM (via litellm) with a strict JSON-output prompt.
   Injects available MCP tools into the prompt for dynamic awareness.
   Falls back to rule-based mode on any failure.

2. Rule-based mode (USE_LLM_PLANNER=false)
   Applies keyword heuristics to produce a sensible default plan.
   Maps keywords to known MCP tools (fetch, read_file, etc.).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import config
from utils.logger import get_logger
from utils.schemas import PlannerOutput, Task

if TYPE_CHECKING:
    from mcp_manager import MCPManager

log = get_logger("Planner")

# ---------------------------------------------------------------------------
# LLM system prompt template
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_TEMPLATE = """
You are a task-planning assistant. The user will give you a high-level goal.
Decompose it into 4–8 concrete, non-overlapping sub-tasks.

Return ONLY a valid JSON object — no markdown, no prose — with this schema:
{{
  "tasks": [
    {{
      "id": "<single uppercase letter or short code>",
      "description": "<what this task does>",
      "depends_on": ["<id of prerequisite task>", ...],
      "required_tools": ["<tool name from the available list>", ...]
    }}
  ],
  "dependencies": [["<from_id>", "<to_id>"], ...]
}}

AVAILABLE TOOLS:
{tools_list}

Rules:
- 'depends_on' lists task ids that MUST complete before this task can start.
- 'dependencies' must match the 'depends_on' fields.
- ONLY use tool names listed in the AVAILABLE TOOLS section.
- If no tool is appropriate for a task, leave 'required_tools' as [].
- The first task must have an empty depends_on list.
- There must be no cycles.
""".strip()


class Planner:
    """Decomposes a user goal into a structured task plan."""

    def __init__(self, mcp_manager: Optional["MCPManager"] = None) -> None:
        self._mcp = mcp_manager

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def plan(self, goal: str) -> PlannerOutput:
        log.info("Planning goal: '%s'", goal)

        if config.USE_LLM_PLANNER:
            try:
                return self._plan_with_llm(goal)
            except Exception as exc:
                log.warning("LLM planner failed (%s); falling back to "
                            "rule-based planner.", exc)

        return self._plan_rule_based(goal)

    # ------------------------------------------------------------------ #
    #  LLM mode                                                            #
    # ------------------------------------------------------------------ #

    def _plan_with_llm(self, goal: str) -> PlannerOutput:
        """Call the configured LLM and parse its JSON response."""
        import litellm

        # Build tools list for the prompt
        tools_desc = "[] (No external tools available)"
        if self._mcp:
            all_tools = self._mcp.get_all_tools()
            if all_tools:
                tools_desc = "\n".join([
                    f"- {t.tool_name}: {t.description}"
                    for t in all_tools
                ])

        log.debug("Calling LLM '%s' for planning …", config.LLM_MODEL)

        prompt = _SYSTEM_PROMPT_TEMPLATE.format(tools_list=tools_desc)

        response = litellm.completion(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY or None,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": f"Goal: {goal}"},
            ],
            temperature=0.2,
        )

        raw: str = response.choices[0].message.content.strip()
        log.debug("LLM raw response:\n%s", raw)

        # Strip possible markdown code fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)

        data: Dict[str, Any] = json.loads(raw)
        return self._parse_raw(data)

    # ------------------------------------------------------------------ #
    #  Rule-based fallback                                                 #
    # ------------------------------------------------------------------ #

    def _plan_rule_based(self, goal: str) -> PlannerOutput:
        """
        Keyword heuristics updated to map to MCP tools.
        """
        log.debug("Using rule-based planner for goal: '%s'", goal)
        goal_lower = goal.lower()

        # Detection logic
        is_web    = any(w in goal_lower for w in ["search", "find", "fetch", "web", "url", "website", "weather"])
        is_file   = any(w in goal_lower for w in ["file", "read", "write", "save", "document"])
        is_think  = any(w in goal_lower for w in ["complex", "step-by-step", "reason", "break down"])

        tasks: List[Task] = []
        deps:  List[List[str]] = []

        # A: Prep
        tasks.append(Task(id="A", description=f"Initial analysis of goal: {goal}", depends_on=[]))

        # B: Research/Data
        if is_web:
            tasks.append(Task(id="B", description="Retrieve external data via web fetch", 
                             depends_on=["A"], required_tools=["fetch"]))
            deps.append(["A", "B"])
            prev = "B"
        elif is_file:
            tasks.append(Task(id="B", description="Access local workspace files", 
                             depends_on=["A"], required_tools=["read_file"]))
            deps.append(["A", "B"])
            prev = "B"
        else:
            prev = "A"

        # C: Processing
        if is_think:
            tasks.append(Task(id="C", description="Reason through findings step-by-step", 
                             depends_on=[prev], required_tools=["sequential_thinking"]))
            deps.append([prev, "C"])
            prev = "C"
        else:
            tasks.append(Task(id="C", description="Process and structure the information", 
                             depends_on=[prev]))
            deps.append([prev, "C"])
            prev = "C"

        # D: Final Output
        if is_file:
            tasks.append(Task(id="D", description="Save the final results to the workspace", 
                             depends_on=[prev], required_tools=["write_file"]))
        else:
            tasks.append(Task(id="D", description="Produce the final summary/answer", 
                             depends_on=[prev]))
        deps.append([prev, "D"])

        plan = PlannerOutput(tasks=tasks, dependencies=deps)
        return plan

    # ------------------------------------------------------------------ #
    #  Parse helper                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_raw(data: Dict[str, Any]) -> PlannerOutput:
        tasks = [
            Task(
                id=t["id"],
                description=t["description"],
                depends_on=t.get("depends_on", []),
                required_tools=t.get("required_tools", []),
            )
            for t in data["tasks"]
        ]
        dependencies: List[List[str]] = data.get("dependencies", [])
        plan = PlannerOutput(tasks=tasks, dependencies=dependencies)
        return plan
