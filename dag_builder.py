"""
dag_builder.py
--------------
DAGBuilder: converts PlannerOutput into an executable DAG.

Features
--------
• Validates no cycles (raises ValueError on cycle detection).
• Topological sort using Kahn's algorithm.
• Groups tasks into "levels" — tasks in the same level have no
  mutual dependency and can be executed in parallel.

Public API
----------
builder = DAGBuilder()
dag     = builder.build(planner_output)

dag.nodes              -> Dict[task_id, Task]
dag.edges              -> Dict[task_id, List[task_id]]  (from -> [to, ...])
dag.levels             -> List[List[task_id]]           (parallel groups)
dag.topological_order  -> List[task_id]                 (flat exec order)
dag.predecessors(id)   -> List[task_id]
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List

from utils.logger import get_logger
from utils.schemas import PlannerOutput, Task

log = get_logger("DAGBuilder")


@dataclass
class DAG:
    """Immutable representation of the task dependency graph."""
    nodes: Dict[str, Task]                      # task_id -> Task
    edges: Dict[str, List[str]]                  # task_id -> [dependents]
    in_degree: Dict[str, int]                    # task_id -> num prerequisites
    levels: List[List[str]] = field(default_factory=list)
    topological_order: List[str] = field(default_factory=list)

    def predecessors(self, task_id: str) -> List[str]:
        """Return all task_ids that must complete before task_id."""
        return self.nodes[task_id].depends_on

    def successors(self, task_id: str) -> List[str]:
        """Return all task_ids that are unblocked when task_id finishes."""
        return self.edges.get(task_id, [])

    def summary(self) -> str:
        lines = [
            f"DAG: {len(self.nodes)} nodes, "
            f"{sum(len(v) for v in self.edges.values())} edges",
            f"Topological order: {' -> '.join(self.topological_order)}",
        ]
        for i, level in enumerate(self.levels):
            lines.append(f"  Level {i}: {level} (can run in parallel)")
        return "\n".join(lines)


class DAGBuilder:
    """Builds and validates a DAG from a PlannerOutput."""

    def build(self, plan: PlannerOutput) -> DAG:
        log.info("Building DAG from %d tasks …", len(plan.tasks))

        # Index tasks
        nodes: Dict[str, Task] = {t.id: t for t in plan.tasks}

        # Build adjacency list (from → list of dependents)
        edges: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int]   = {tid: 0 for tid in nodes}

        for task in plan.tasks:
            for dep in task.depends_on:
                if dep not in nodes:
                    raise ValueError(
                        f"Task '{task.id}' depends on unknown task '{dep}'"
                    )
                edges[dep].append(task.id)
                in_degree[task.id] += 1

        # Kahn's algorithm — topological sort + cycle detection
        queue: deque[str] = deque(
            tid for tid, deg in in_degree.items() if deg == 0
        )
        topo_order: List[str] = []
        levels: List[List[str]] = []

        # Track in-degree copy for level assignment
        in_deg_copy = dict(in_degree)

        while queue:
            # All nodes currently in the queue form one parallel level
            current_level = list(queue)
            levels.append(current_level)
            queue.clear()

            for tid in current_level:
                topo_order.append(tid)
                for successor in edges.get(tid, []):
                    in_deg_copy[successor] -= 1
                    if in_deg_copy[successor] == 0:
                        queue.append(successor)

        if len(topo_order) != len(nodes):
            cycle_nodes = set(nodes) - set(topo_order)
            raise ValueError(
                f"Cycle detected in DAG involving tasks: {cycle_nodes}"
            )

        dag = DAG(
            nodes=nodes,
            edges=dict(edges),
            in_degree=in_degree,
            levels=levels,
            topological_order=topo_order,
        )

        log.info("DAG built successfully.\n%s", dag.summary())
        return dag
