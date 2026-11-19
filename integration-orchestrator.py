import asyncio
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from fastmcp import FastMCP

from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:
    AsyncSqliteSaver = None

from orchestrator_core import (
    ArtifactFile,
    build_scrubbed_env,
    extract_and_write_files,
    get_source_snapshot,
    get_sqlite_checkpointer,
    get_workspace_lock,
    log_audit_event,
    validate_python_ast,
    get_llm
)

# =====================================================================
# 1. INITIALIZE FASTMCP SERVER & MODEL
# =====================================================================

mcp = FastMCP("SDLC-Integration-E2E-Orchestrator")

llm = get_llm()
#llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0)
# llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.0, num_ctx=4096)


# =====================================================================
# 2. STATE & SCHEMAS
# =====================================================================

class IntegrationState(TypedDict):
    workspace_path: str
    compose_project_name: str
    arch_text: str
    codebase_snapshot: str
    written_files: List[ArtifactFile]

    last_stdout: str
    last_stderr: str
    last_exit_code: int
    iteration_count: int


# =====================================================================
# 3. AGENT NODES
# =====================================================================

def integration_architect_agent(state: IntegrationState):
    sys.stderr.write("\n[Phase 4 -> Agent 1] Generating E2E test suite & isolated Docker Compose config...\n")
    workspace = Path(state["workspace_path"])
    fence = "`" * 3

    retry_note = ""
    if state.get("iteration_count", 0) > 0 and state.get("last_exit_code", 0) != 0:
        last_written = "\n\n".join(f"### {f.file_path}\n{f.content}" for f in state.get("written_files", []))
        retry_note = (
            f"\nPREVIOUS INTEGRATION RUN FAILED (Exit Code {state['last_exit_code']}):\n"
            f"Stdout: {state['last_stdout'][-1500:]}\n"
            f"Stderr: {state['last_stderr'][-1500:]}\n"
            f"Files written previously:\n{last_written[:2000]}\n"
            f"Fix any service port mismatches, healthcheck timing, or pytest assertion issues.\n"
        )

    prompt = f"""
    You are a Principal Distributed Systems & Integration QA Architect.
    Generate an isolated End-to-End (E2E) integration test suite and non-destructive test environment.

    [Architecture Context]
    {state['arch_text'][:2500]}

    [Application Source Code]
    {state['codebase_snapshot']}
    {retry_note}

    Mandatory Deliverables:
    1. `tests/integration/test_e2e_workflows.py`: End-to-end pytest tests exercising multi-component flows, concurrency, and persistence. Import modules from `src/`.
    2. `docker-compose.test.yml`: Isolated service dependencies (e.g., Redis/Postgres if needed by architecture) with healthchecks and ephemeral volumes (no host port conflicts).
    3. `.env.test`: Isolated test environment variables.

    Output EVERY file strictly using this Markdown format (do NOT output JSON):

    FILE: tests/integration/test_e2e_workflows.py
    {fence}python
    # complete multi-line E2E pytest suite here
    {fence}

    FILE: docker-compose.test.yml
    {fence}yaml
    # isolated compose services here
    {fence}

    FILE: .env.test
    {fence}env
    APP_ENV=test
    {fence}
    """
    res = llm.invoke(prompt)
    written = extract_and_write_files(workspace, res.content)
    return {"written_files": written}


def e2e_sandbox_runner(state: IntegrationState):
    workspace = Path(state["workspace_path"])
    proj_name = state["compose_project_name"]
    new_iter = state.get("iteration_count", 0) + 1
    sys.stderr.write(f"\n[Phase 4 -> Runner] Executing E2E suite Pass #{new_iter} (project: {proj_name})...\n")

    syntax_err = validate_python_ast(state.get("written_files", []))
    if syntax_err:
        sys.stderr.write(f"[Phase 4 -> AST Gate] {syntax_err}\n")
        return {
            "last_stdout": "",
            "last_stderr": syntax_err,
            "last_exit_code": 2,
            "iteration_count": new_iter,
        }

    env = build_scrubbed_env(workspace)
    compose_file = workspace / "docker-compose.test.yml"
    has_compose = compose_file.exists() and "services:" in compose_file.read_text(encoding="utf-8", errors="ignore")

    try:
        if has_compose:
            up_proc = subprocess.run(
                ["docker", "compose", "-p", proj_name, "-f", "docker-compose.test.yml", "up", "-d", "--wait"],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
            if up_proc.returncode != 0:
                return {
                    "last_stdout": up_proc.stdout,
                    "last_stderr": f"Docker Compose startup failed: {up_proc.stderr}",
                    "last_exit_code": up_proc.returncode,
                    "iteration_count": new_iter,
                }

        test_proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/integration", "-v", "--tb=short", "--maxfail=1"],
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout, stderr, code = test_proc.stdout, test_proc.stderr, test_proc.returncode
    except Exception as e:
        stdout, stderr, code = "", str(e), 1
    finally:
        if has_compose:
            subprocess.run(
                ["docker", "compose", "-p", proj_name, "-f", "docker-compose.test.yml", "down", "-v", "--remove-orphans"],
                cwd=workspace,
                env=env,
                capture_output=True,
            )

    sys.stderr.write(f"[Phase 4 -> Runner] Pass #{new_iter} Exit Code: {code}\n")
    return {
        "last_stdout": stdout,
        "last_stderr": stderr,
        "last_exit_code": code,
        "iteration_count": new_iter,
    }


# =====================================================================
# 4. GRAPH COMPILATION HELPER
# =====================================================================

def route_e2e(state: IntegrationState):
    if state.get("last_exit_code", 1) == 0:
        return "done"
    if state.get("iteration_count", 0) >= 3:
        sys.stderr.write("[Phase 4] Max E2E repair attempts (3) reached.\n")
        return "done"
    return "retry"


def build_integration_graph(checkpointer=None):
    workflow = StateGraph(IntegrationState)
    workflow.add_node("integration_architect", integration_architect_agent)
    workflow.add_node("e2e_runner", e2e_sandbox_runner)

    workflow.add_edge(START, "integration_architect")
    workflow.add_edge("integration_architect", "e2e_runner")
    workflow.add_conditional_edges("e2e_runner", route_e2e, {"done": END, "retry": "integration_architect"})
    return workflow.compile(checkpointer=checkpointer)


# =====================================================================
# 5. MCP TOOL REGISTRATION
# =====================================================================

@mcp.tool()
async def verify_e2e_integration(
    run_id: str = "",
    workspace_path: str = ".",
) -> str:
    """
    Executes Phase 4 (Integration & E2E Verification) with isolated Docker Compose project
    namespacing, guaranteed teardown cleanup, AST gates, and SQLite checkpointing.
    """
    root = Path(workspace_path).resolve()
    unique_id = run_id.strip() or uuid.uuid4().hex[:8]
    safe_proj = f"sdlc_e2e_{re.sub(r'[^a-z0-9]', '', unique_id.lower())}"
    thread_id = f"phase4-{unique_id}"

    async with get_workspace_lock(root):
        arch_file = root / "docs" / "architecture.md"
        initial_state: IntegrationState = {
            "workspace_path": str(root),
            "compose_project_name": safe_proj,
            "arch_text": arch_file.read_text(encoding="utf-8") if arch_file.exists() else "",
            "codebase_snapshot": get_source_snapshot(root),
            "written_files": [],
            "last_stdout": "",
            "last_stderr": "",
            "last_exit_code": 1,
            "iteration_count": 0,
        }

        config = {"configurable": {"thread_id": thread_id}}
        if AsyncSqliteSaver is not None:
            os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
            db_path = root / ".orchestrator" / "checkpoints.sqlite"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                graph = build_integration_graph(checkpointer=saver)
                final_state = await graph.ainvoke(initial_state, config=config)
        else:
            graph = build_integration_graph(checkpointer=get_sqlite_checkpointer(root))
            final_state = await asyncio.to_thread(graph.invoke, initial_state, config)

        passed = final_state.get("last_exit_code", 1) == 0
        iters = final_state.get("iteration_count", 0)
        status = "E2E_VERIFIED" if passed else "E2E_FAILED_MAX_RETRIES"

        log_audit_event(
            workspace=root,
            phase="phase4_integration",
            thread_id=thread_id,
            status=status,
            iterations=iters,
            details={"compose_project": safe_proj},
        )

        return f"Phase 4 Integration {status} in {iters} pass(es) (project namespace: {safe_proj})."


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "http":
        mcp.run(
            transport="streamable-http",
            host=os.getenv("MCP_HOST", "127.0.0.1"),
            port=int(os.getenv("MCP_PORT", "8004")),
        )
    else:
        mcp.run()