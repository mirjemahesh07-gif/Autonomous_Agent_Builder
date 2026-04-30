"""
agent_factory.py
----------------
AgentFactory: dynamically creates Agent instances, one per task.

Each Agent
----------
• Holds an AgentConfig (task + resolved tools + upstream inputs).
• Delegates execution to the ExecutionEngine.
• Returns a structured AgentOutput.

The factory itself
------------------
• Accepts the DAG and Tool Resolver.
• Resolves all required tools before creating the agent.
• Returns a Dict[task_id, Agent] ready for the Orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from utils.schemas import (
    AgentConfig, AgentOutput, Task, TaskStatus, ToolDefinition
)

log = get_logger("AgentFactory")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    A self-contained execution unit for one task.
    Instantiated by AgentFactory; executed by the Orchestrator via
    ExecutionEngine.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config    = config
        self.task      = config.task
        self.tools     = config.tools
        self.task_id   = config.task.id
        self._engine   = None   # injected by Orchestrator at runtime

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def set_inputs(self, inputs: Dict[str, Any]) -> None:
        """Update upstream context inputs before execution."""
        self.config.inputs.update(inputs)

    async def run(self, inputs: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """
        Execute the task.  The Orchestrator calls this; internally it
        delegates to the ExecutionEngine if one is attached, otherwise
        it runs a simple built-in simulation.
        """
        if inputs:
            self.set_inputs(inputs)

        if self._engine is not None:
            return await self._engine.execute(self, self.config.inputs)

        # Fallback: pure simulation (no engine injected)
        log.debug("Agent %s running in standalone simulation mode.", self.task_id)
        return self._simulate()

    # ------------------------------------------------------------------ #
    #  Simulation                                                          #
    # ------------------------------------------------------------------ #

    def _simulate(self) -> AgentOutput:
        """Produce a plausible simulated result (no real I/O)."""
        tool_names = [t.name for t in self.tools]
        result = {
            "summary": f"[SIMULATED] Completed: {self.task.description}",
            "inputs_received": list(self.config.inputs.keys()),
            "tools_available": tool_names,
        }
        return AgentOutput(
            task_id=self.task_id,
            result=result,
            status=TaskStatus.COMPLETED,
            used_tools=tool_names,
        )

    # ------------------------------------------------------------------ #
    #  Repr                                                                #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.task_id!r}, "
            f"task={self.task.description[:40]!r}, "
            f"tools={[t.name for t in self.tools]})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class AgentFactory:
    """Creates and configures one Agent per task node in the DAG."""

    def __init__(self, tool_resolver=None) -> None:
        """
        Parameters
        ----------
        tool_resolver : ToolResolver | None
            If provided, required tools are resolved at factory time.
        """
        self._resolver = tool_resolver

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def create_agents(self, tasks: List[Task]) -> Dict[str, Agent]:
        """
        Create one Agent per task.

        Returns
        -------
        Dict[task_id, Agent]
        """
        agents: Dict[str, Agent] = {}
        for task in tasks:
            agent = self._create_one(task)
            agents[task.id] = agent
            log.info("Created %r", agent)
        return agents

    def create_agent(self, task: Task) -> Agent:
        """Create a single Agent for a given task."""
        return self._create_one(task)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _create_one(self, task: Task) -> Agent:
        tools: List[ToolDefinition] = []

        if self._resolver and task.required_tools:
            for tool_name in task.required_tools:
                try:
                    tool_def = self._resolver.resolve(tool_name)
                    tools.append(tool_def)
                    log.debug("Tool '%s' resolved for task '%s'.",
                              tool_name, task.id)
                except Exception as exc:
                    log.warning("Could not resolve tool '%s' for task '%s': %s",
                                tool_name, task.id, exc)

        cfg = AgentConfig(task=task, tools=tools, inputs={})
        return Agent(cfg)
