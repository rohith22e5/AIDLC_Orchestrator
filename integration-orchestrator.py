import asyncio
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import List
from typing_extensions import TypedDict
from fastmcp import FastMCP

from langgraph.graph import StateGraph, START, END

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:
    AsyncSqliteSaver = None

from orchestrator_core import (
    ArtifactFile,
    TokenBudgetExceededError,
    build_scrubbed_env,
    extract_and_write_files,
    get_llm,
    get_source_snapshot,
    get_sqlite_checkpointer,
    get_workspace_lock,
    load_agent_prompt,
    log_audit_event,
    validate_python_ast,
)

mcp = FastMCP("SDLC-Integration-E2E-Orchestrator")


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


def integration_architect_agent(state: IntegrationState):
    sys.stderr.write("\n[Phase 4 -> Agent 1] Generating E2E test suite & isolated Docker Compose config...\n")
    workspace = Path(state["workspace_path"])
    llm = get_llm()
    is_retry = state.get("iteration_count", 0) > 0 and state.get("last_exit_code", 0) != 0

    system_spec = load_agent_prompt(
        "integration_architect",
        ["python_validation_contracts", "docker_devops_contracts", "markdown_file_output"],
        workspace=workspace,
        include_retry_skill=is_retry,
    )

    retry_note = ""
    if is_retry:
        last_written = "\n\n".join(f"### {f.file_path}\n{f.content}" for f in state.get("written_files", []))
        retry_note = (
            f"\n[PREVIOUS INTEGRATION RUN FAILED (Exit Code {state['last_exit_code']})]\n"
            f"Stdout: {state['last_stdout'][-1500:]}\n"
            f"Stderr: {state['last_stderr'][-1500:]}\n"
            f"Files written previously:\n{last_written[:2000]}\n"
        )

    prompt = f"""{system_spec}

[Architecture Context]
{state['arch_text'][:2500]}

[Application Source Code]
{state['codebase_snapshot']}
{retry_note}"""

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
        return {"last_stdout": "", "last_stderr": syntax_err, "last_exit_code": 2, "iteration_count": new_iter}

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

    return {"last_stdout": stdout, "last_stderr": stderr, "last_exit_code": code, "iteration_count": new_iter}


def route_e2e(state: IntegrationState):
    if state.get("last_exit_code", 1) == 0 or state.get("iteration_count", 0) >= 3:
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


@mcp.tool()
async def verify_e2e_integration(
    run_id: str = "",
    workspace_path: str = ".",
) -> str:
    """Executes Phase 4 (Integration & E2E Verification) using dynamic agents.md & skills.md specs."""
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
        try:
            if AsyncSqliteSaver is not None:
                os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
                db_path = root / ".orchestrator" / "checkpoints.sqlite"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                    final_state = await build_integration_graph(checkpointer=saver).ainvoke(initial_state, config=config)
            else:
                final_state = await asyncio.to_thread(
                    build_integration_graph(checkpointer=get_sqlite_checkpointer(root)).invoke, initial_state, config
                )
        except TokenBudgetExceededError as e:
            log_audit_event(root, "phase4_integration", thread_id, "TOKEN_BUDGET_HALTED", 0, {"error": str(e)})
            return f"Phase 4 HALTED by token circuit breaker: {e}"

        passed = final_state.get("last_exit_code", 1) == 0
        iters = final_state.get("iteration_count", 0)
        status = "E2E_VERIFIED" if passed else "E2E_FAILED_MAX_RETRIES"

        log_audit_event(root, "phase4_integration", thread_id, status, iters, {"compose_project": safe_proj})
        return f"Phase 4 Integration {status} in {iters} pass(es) (project namespace: {safe_proj})."


if __name__ == "__main__":
    if os.getenv("MCP_TRANSPORT", "stdio").lower() == "http":
        mcp.run(transport="streamable-http", host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8004")))
    else:
        mcp.run()