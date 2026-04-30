"""
config.py
---------
System-wide configuration for the Agentic DAG Orchestration System.
All tunable knobs live here — never hardcode these inside modules.
"""

import os

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash")
LLM_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# If True the planner uses an LLM; if False it uses the rule-based fallback.
USE_LLM_PLANNER: bool = os.getenv("USE_LLM_PLANNER", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Orchestrator settings
# ---------------------------------------------------------------------------
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_SECONDS: float = float(os.getenv("RETRY_BACKOFF_SECONDS", "2.0"))
PARALLEL_EXECUTION: bool = os.getenv("PARALLEL_EXECUTION", "true").lower() == "true"
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))

# ---------------------------------------------------------------------------
# Tool Registry paths
# ---------------------------------------------------------------------------
TOOL_REGISTRY_ROOT: str = os.path.join(
    os.path.dirname(__file__), "tool_registry"
)
TOOL_REGISTRY_OPENAPI_DIR: str = os.path.join(TOOL_REGISTRY_ROOT, "openapi")
TOOL_REGISTRY_MCP_DIR: str = os.path.join(TOOL_REGISTRY_ROOT, "mcp")
TOOL_REGISTRY_METADATA_DIR: str = os.path.join(TOOL_REGISTRY_ROOT, "metadata")

# ---------------------------------------------------------------------------
# MCP settings
# ---------------------------------------------------------------------------
MCP_SERVERS_CONFIG: str = os.path.join(os.path.dirname(__file__), "mcp_servers.json")
MCP_CONNECT_TIMEOUT: int = int(os.getenv("MCP_CONNECT_TIMEOUT", "30"))
MCP_TOOL_TIMEOUT: int = int(os.getenv("MCP_TOOL_TIMEOUT", "30"))
MCP_ENABLED: bool = os.getenv("MCP_ENABLED", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
