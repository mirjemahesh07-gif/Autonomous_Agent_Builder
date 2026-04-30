"""
main.py
-------
Entry point for the Agentic DAG Orchestration System (Async / MCP Enabled).

Usage
-----
# Run with a sample goal (default):
    python main.py

# Run with a custom goal:
    python main.py "Fetch and summarize the content from https://python.org"
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

# ── Core modules ──────────────────────────────────────────────────────────
from planner          import Planner
from dag_builder      import DAGBuilder
from agent_factory    import AgentFactory
from orchestrator     import Orchestrator
from tool_resolver    import ToolResolver
from execution_engine import ExecutionEngine
from mcp_manager      import MCPManager
from tool_registry.registry import ToolRegistry
from utils.logger     import get_logger
import config

log = get_logger("Main")

# ── Default sample goal ────────────────────────────────────────────────────
DEFAULT_GOAL = (
    "Fetch the content from https://modelcontextprotocol.io, "
    "break down its core concepts step-by-step, and save the summary to a file."
)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _banner(goal: str) -> None:
    width = 68
    log.info("┌" + "─" * width + "┐")
    log.info("│%s│", "  Agentic DAG Orchestration System (MCP)".center(width))
    log.info("├" + "─" * width + "┤")
    for chunk in [goal[i:i+width-4] for i in range(0, len(goal), width-4)]:
        log.info("│  %-*s  │", width - 4, chunk)
    log.info("└" + "─" * width + "┘")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(goal: str) -> dict:
    """
    Execute the full agentic pipeline for a given goal.
    """
    start = time.perf_counter()
    _banner(goal)

    # ── 0. MCP Initialization ──────────────────────────────────────────
    mcp_manager = MCPManager()
    if config.MCP_ENABLED:
        log.info("\n[Step 0/5] Initializing MCP Servers …")
        await mcp_manager.connect_all()

    try:
        # ── 1. Plan ──────────────────────────────────────────────────────────
        log.info("\n[Step 1/5] Planning …")
        planner      = Planner(mcp_manager)
        planner_out  = planner.plan(goal)

        # ── 2. Build DAG ──────────────────────────────────────────────────────
        log.info("\n[Step 2/5] Building DAG …")
        dag_builder  = DAGBuilder()
        dag          = dag_builder.build(planner_out)

        # ── 3. Resolve tools & create agents ──────────────────────────────────
        log.info("\n[Step 3/5] Resolving tools & creating agents …")
        registry     = ToolRegistry()
        resolver     = ToolResolver(registry, mcp_manager)
        factory      = AgentFactory(tool_resolver=resolver)
        agents       = factory.create_agents(planner_out.tasks)

        # ── 4. Execute ────────────────────────────────────────────────────────
        log.info("\n[Step 4/5] Executing DAG …")
        engine       = ExecutionEngine(mcp_manager)
        orchestrator = Orchestrator(engine)
        results      = await orchestrator.run(dag, agents)

        # ── 5. Summarise ──────────────────────────────────────────────────────
        elapsed = time.perf_counter() - start
        log.info("\n[Step 5/5] Pipeline complete in %.2fs", elapsed)

        summary = _build_summary(goal, dag, results, elapsed)
        _print_summary(summary)
        return summary

    finally:
        if config.MCP_ENABLED:
            await mcp_manager.disconnect_all()


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _build_summary(goal: str, dag, results, elapsed: float) -> dict:
    task_summaries = {}
    for tid, output in results.items():
        task_summaries[tid] = output.to_dict()

    # Extract final output
    final_output = None
    for tid in reversed(dag.topological_order):
        out = results.get(tid)
        if out and out.result:
            final_output = out.result
            break

    return {
        "goal":           goal,
        "total_tasks":    len(dag.nodes),
        "elapsed_seconds": round(elapsed, 3),
        "execution_order": dag.topological_order,
        "parallel_levels": dag.levels,
        "task_results":   task_summaries,
        "final_output":   final_output,
    }


def _print_summary(summary: dict) -> None:
    width = 68
    log.info("\n" + "═" * (width + 2))
    log.info("  FINAL PIPELINE SUMMARY")
    log.info("═" * (width + 2))
    log.info("Goal         : %s", summary["goal"][:width])
    log.info("Total tasks  : %d", summary["total_tasks"])
    log.info("Elapsed      : %.3fs", summary["elapsed_seconds"])
    log.info("Exec order   : %s", " → ".join(summary["execution_order"]))

    log.info("\n  Task Results:")
    for tid, res in summary["task_results"].items():
        icon = {"completed": "✔", "failed": "✘", "skipped": "⊘"}.get(
            res["status"], "?"
        )
        log.info("  %s  [%s] %s", icon, tid, res["status"].upper())
        if res.get("used_tools"):
            log.info("       Tools used: %s", res["used_tools"])

    log.info("\n  Final Output:")
    if summary["final_output"]:
        output_str = json.dumps(summary["final_output"], indent=4)
        for line in output_str.splitlines():
            log.info("  %s", line)
    else:
        log.info("  (no output produced)")

    log.info("═" * (width + 2))


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_GOAL
    try:
        asyncio.run(run_pipeline(goal))
    except KeyboardInterrupt:
        log.info("\nPipeline interrupted by user.")
    except Exception as e:
        log.exception("Pipeline failed with error: %s", e)
