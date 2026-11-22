import asyncio
import os
import shutil
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
)

mcp = FastMCP("SDLC-DevOps-CICD-Orchestrator")


class DevOpsState(TypedDict):
    workspace_path: str
    arch_text: str
    codebase_snapshot: str
    written_files: List[ArtifactFile]
    validation_stderr: str
    validation_exit_code: int
    iteration_count: int


def devops_engineer_agent(state: DevOpsState):
    sys.stderr.write("\n[Phase 5 -> Agent 1] Generating production Dockerfile, Compose & GitHub Actions CI...\n")
    workspace = Path(state["workspace_path"])
    llm = get_llm()
    is_retry = state.get("iteration_count", 0) > 0 and state.get("validation_exit_code", 0) != 0

    system_spec = load_agent_prompt(
        "devops_engineer",
        ["docker_devops_contracts", "markdown_file_output"],
        workspace=workspace,
        include_retry_skill=is_retry,
    )

    retry_note = (
        f"\n[PREVIOUS CONFIG VALIDATION FAILED (Exit Code {state['validation_exit_code']})]\n"
        f"{state['validation_stderr'][-1500:]}\n"
        if is_retry
        else ""
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


def devops_config_validator(state: DevOpsState):
    workspace = Path(state["workspace_path"])
    new_iter = state.get("iteration_count", 0) + 1
    sys.stderr.write(f"\n[Phase 5 -> Validator] Validating DevOps configs Pass #{new_iter}...\n")

    env = build_scrubbed_env(workspace)
    errors = []

    try:
        import yaml  # type: ignore
        for f in state.get("written_files", []):
            if f.file_path.endswith((".yml", ".yaml")):
                yaml.safe_load(f.content)
    except ImportError:
        pass
    except Exception as e:
        errors.append(f"YAML parse error: {e}")

    compose_file = workspace / "docker-compose.yml"
    docker_exe = shutil.which("docker")
    if compose_file.exists() and docker_exe:
        proc = subprocess.run(
            [docker_exe, "compose", "-f", "docker-compose.yml", "config", "--quiet"],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode != 0:
            errors.append(f"docker compose config failed: {proc.stderr}")

    return {"validation_stderr": "\n".join(errors), "validation_exit_code": 1 if errors else 0, "iteration_count": new_iter}


def route_devops(state: DevOpsState):
    if state.get("validation_exit_code", 1) == 0 or state.get("iteration_count", 0) >= 3:
        return "done"
    return "retry"


def build_devops_graph(checkpointer=None):
    workflow = StateGraph(DevOpsState)
    workflow.add_node("devops_engineer", devops_engineer_agent)
    workflow.add_node("validator", devops_config_validator)

    workflow.add_edge(START, "devops_engineer")
    workflow.add_edge("devops_engineer", "validator")
    workflow.add_conditional_edges("validator", route_devops, {"done": END, "retry": "devops_engineer"})
    return workflow.compile(checkpointer=checkpointer)


@mcp.tool()
async def generate_devops_pipeline(
    deploy_id: str = "",
    workspace_path: str = ".",
) -> str:
    """Executes Phase 5 of SDLC using dynamic agents.md & skills.md specs."""
    root = Path(workspace_path).resolve()
    uid = deploy_id.strip() or uuid.uuid4().hex[:8]
    thread_id = f"phase5-{uid}"

    async with get_workspace_lock(root):
        arch_file = root / "docs" / "architecture.md"
        initial_state: DevOpsState = {
            "workspace_path": str(root),
            "arch_text": arch_file.read_text(encoding="utf-8") if arch_file.exists() else "",
            "codebase_snapshot": get_source_snapshot(root),
            "written_files": [],
            "validation_stderr": "",
            "validation_exit_code": 1,
            "iteration_count": 0,
        }

        config = {"configurable": {"thread_id": thread_id}}
        try:
            if AsyncSqliteSaver is not None:
                os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
                db_path = root / ".orchestrator" / "checkpoints.sqlite"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                    final_state = await build_devops_graph(checkpointer=saver).ainvoke(initial_state, config=config)
            else:
                final_state = await asyncio.to_thread(
                    build_devops_graph(checkpointer=get_sqlite_checkpointer(root)).invoke, initial_state, config
                )
        except TokenBudgetExceededError as e:
            log_audit_event(root, "phase5_devops", thread_id, "TOKEN_BUDGET_HALTED", 0, {"error": str(e)})
            return f"Phase 5 HALTED by token circuit breaker: {e}"

        passed = final_state.get("validation_exit_code", 1) == 0
        iters = final_state.get("iteration_count", 0)
        status = "DEVOPS_VALIDATED" if passed else "DEVOPS_GENERATED_WITH_WARNINGS"

        log_audit_event(root, "phase5_devops", thread_id, status, iters)
        return f"Phase 5 DevOps {status} in {iters} pass(es). Generated Dockerfile, docker-compose.yml, and .github/workflows/ci.yml."


if __name__ == "__main__":
    if os.getenv("MCP_TRANSPORT", "stdio").lower() == "http":
        mcp.run(transport="streamable-http", host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8005")))
    else:
        mcp.run()