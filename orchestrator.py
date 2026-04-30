"""
orchestrator.py
---------------
Orchestrator: executes a DAG of agents in the correct dependency order.

Execution model
---------------
• Processes the DAG level by level (each level = independent parallel group).
• Within a level, tasks run concurrently via asyncio.gather if
  PARALLEL_EXECUTION=true, otherwise sequentially.
• Passes the outputs of completed tasks as 'inputs' to their dependents.
• Tracks state: pending | running | completed | failed.
• On failure, marks dependents as skipped.

Public API
----------
orchestrator = Orchestrator(engine)
results = await orchestrator.run(dag, agents)   -> Dict[task_id, AgentOutput]
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import config
from execution_engine import ExecutionEngine
from utils.logger import get_logger
from utils.schemas import AgentOutput, TaskStatus

log = get_logger("Orchestrator")


class Orchestrator:
    """Level-by-level DAG executor with async support."""

    def __init__(self, engine: Optional[ExecutionEngine] = None) -> None:
        self._engine = engine or ExecutionEngine()

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    async def run(self, dag, agents: Dict[str, Any]) -> Dict[str, AgentOutput]:
        """
        Execute all agents according to the DAG.

        Parameters
        ----------
        dag     : DAG            (from dag_builder.DAGBuilder)
        agents  : Dict[str, Agent]

        Returns
        -------
        Dict[task_id, AgentOutput]
        """
        log.info("=" * 60)
        log.info("Orchestrator starting — %d tasks, %d levels.",
                 len(dag.nodes), len(dag.levels))
        log.info("=" * 60)

        # Inject engine into every agent
        for agent in agents.values():
            agent._engine = self._engine

        results:     Dict[str, AgentOutput] = {}
        failed_ids:  set = set()

        for level_idx, level in enumerate(dag.levels):
            log.info("── Level %d: %s", level_idx, level)

            # Separate tasks into runnable vs skipped
            runnable, skipped = [], []
            for tid in level:
                preds = dag.predecessors(tid)
                blocked = [p for p in preds if p in failed_ids]
                if blocked:
                    skipped.append((tid, blocked))
                else:
                    runnable.append(tid)

            # Mark skipped tasks
            for tid, blocked_by in skipped:
                log.warning("  Skipping task '%s' — predecessor(s) failed: %s",
                            tid, blocked_by)
                results[tid] = AgentOutput(
                    task_id=tid,
                    result=None,
                    status=TaskStatus.SKIPPED,
                    error=f"Skipped because {blocked_by} failed",
                )
                failed_ids.add(tid)

            if not runnable:
                continue

            # Execute level concurrently or sequentially
            level_results = await self._execute_level(
                runnable, agents, dag, results
            )
            results.update(level_results)

            # Record failures
            for tid, output in level_results.items():
                if output.status == TaskStatus.FAILED:
                    failed_ids.add(tid)

        log.info("=" * 60)
        log.info("Orchestrator finished. Summary:")
        for tid, out in results.items():
            icon = {"completed": "✔", "failed": "✘",
                    "skipped": "⊘"}.get(out.status.value, "?")
            log.info("  %s  Task '%s': %s", icon, tid, out.status.value)
        log.info("=" * 60)

        return results

    # ------------------------------------------------------------------ #
    #  Level execution                                                     #
    # ------------------------------------------------------------------ #

    async def _execute_level(
        self,
        task_ids: List[str],
        agents: Dict[str, Any],
        dag,
        completed_results: Dict[str, AgentOutput],
    ) -> Dict[str, AgentOutput]:
        """Execute one DAG level (parallel or sequential)."""
        if config.PARALLEL_EXECUTION and len(task_ids) > 1:
            return await self._execute_parallel(
                task_ids, agents, dag, completed_results
            )
        return await self._execute_sequential(
            task_ids, agents, dag, completed_results
        )

    async def _execute_sequential(
        self,
        task_ids: List[str],
        agents: Dict[str, Any],
        dag,
        completed_results: Dict[str, AgentOutput],
    ) -> Dict[str, AgentOutput]:
        level_results: Dict[str, AgentOutput] = {}
        for tid in task_ids:
            inputs = self._gather_inputs(tid, dag, completed_results)
            output = await agents[tid].run(inputs)
            level_results[tid] = output
        return level_results

    async def _execute_parallel(
        self,
        task_ids: List[str],
        agents: Dict[str, Any],
        dag,
        completed_results: Dict[str, AgentOutput],
    ) -> Dict[str, AgentOutput]:
        log.debug("  Running %d tasks in parallel via asyncio …", len(task_ids))
        
        # Build tasks list
        tasks = []
        for tid in task_ids:
            inputs = self._gather_inputs(tid, dag, completed_results)
            tasks.append(agents[tid].run(inputs))
            
        # Run all concurrently
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        level_results: Dict[str, AgentOutput] = {}
        for tid, output in zip(task_ids, outputs):
            if isinstance(output, Exception):
                log.error("Task '%s' raised an unexpected exception: %s",
                          tid, output)
                output = AgentOutput(
                    task_id=tid,
                    result=None,
                    status=TaskStatus.FAILED,
                    error=str(output),
                )
            level_results[tid] = output
            
        return level_results

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gather_inputs(
        task_id: str,
        dag,
        completed_results: Dict[str, AgentOutput],
    ) -> Dict[str, Any]:
        """Aggregate outputs of all predecessor tasks as inputs."""
        inputs: Dict[str, Any] = {}
        for pred_id in dag.predecessors(task_id):
            pred_out = completed_results.get(pred_id)
            if pred_out and pred_out.result:
                inputs[pred_id] = pred_out.result
        return inputs
