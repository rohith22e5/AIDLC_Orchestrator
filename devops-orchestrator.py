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
    get_llm
)

# =====================================================================
# 1. INITIALIZE FASTMCP SERVER & MODEL
# =====================================================================

mcp = FastMCP("SDLC-DevOps-CICD-Orchestrator")

llm = get_llm()
#llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0)
# llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.0, num_ctx=4096)


# =====================================================================
# 2. STATE SCHEMA
# =====================================================================

class DevOpsState(TypedDict):
    workspace_path: str
    arch_text: str
    codebase_snapshot: str
    written_files: List[ArtifactFile]
    validation_stderr: str
    validation_exit_code: int
    iteration_count: int


# =====================================================================
# 3. AGENT & VALIDATOR NODES
# =====================================================================

def devops_engineer_agent(state: DevOpsState):
    sys.stderr.write("\n[Phase 5 -> Agent 1] Generating production Dockerfile, Compose & GitHub Actions CI...\n")
    workspace = Path(state["workspace_path"])
    fence = "`" * 3

    retry_note = ""
    if state.get("iteration_count", 0) > 0 and state.get("validation_exit_code", 0) != 0:
        retry_note = (
            f"\nPREVIOUS CONFIG VALIDATION FAILED (Exit Code {state['validation_exit_code']}):\n"
            f"{state['validation_stderr'][-1500:]}\n"
            f"Fix YAML/Dockerfile syntax or invalid Compose schema fields.\n"
        )

    prompt = f"""
    You are a Principal DevOps & Platform Engineer.
    Generate hardened containerization and CI/CD artifacts for the project below.

    [Architecture Context]
    {state['arch_text'][:2500]}

    [Application Source Code]
    {state['codebase_snapshot']}
    {retry_note}

    Mandatory Deliverables:
    1. `Dockerfile`: Multi-stage slim Python image running as a non-root user (`appuser`) with `HEALTHCHECK`.
    2. `docker-compose.yml`: Production-ready service definition with restart policy, resource limits, and healthcheck.
    3. `.github/workflows/ci.yml`: GitHub Actions CI pipeline running `ruff`, `pytest -v`, and container build check on push/PR.

    Output EVERY file strictly using this Markdown format (do NOT output JSON):

    FILE: Dockerfile
    {fence}dockerfile
    # multi-stage Dockerfile here
    {fence}

    FILE: docker-compose.yml
    {fence}yaml
    # compose YAML here
    {fence}

    FILE: .github/workflows/ci.yml
    {fence}yaml
    # GitHub Actions workflow YAML here
    {fence}
    """
    res = llm.invoke(prompt)
    written = extract_and_write_files(workspace, res.content)
    return {"written_files": written}


def devops_config_validator(state: DevOpsState):
    workspace = Path(state["workspace_path"])
    new_iter = state.get("iteration_count", 0) + 1
    sys.stderr.write(f"\n[Phase 5 -> Validator] Validating DevOps configs Pass #{new_iter}...\n")

    env = build_scrubbed_env(workspace)
    errors = []

    # Validate YAML syntax via PyYAML if installed
    try:
        import yaml  # type: ignore
        for f in state.get("written_files", []):
            if f.file_path.endswith((".yml", ".yaml")):
                yaml.safe_load(f.content)
    except ImportError:
        pass
    except Exception as e:
        errors.append(f"YAML parse error: {e}")

    # Validate docker-compose.yml schema using Docker CLI if available
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

    err_text = "\n".join(errors)
    code = 1 if errors else 0
    sys.stderr.write(f"[Phase 5 -> Validator] Exit Code: {code}\n")
    return {
        "validation_stderr": err_text,
        "validation_exit_code": code,
        "iteration_count": new_iter,
    }


# =====================================================================
# 4. GRAPH COMPILATION HELPER
# =====================================================================

def route_devops(state: DevOpsState):
    if state.get("validation_exit_code", 1) == 0:
        return "done"
    if state.get("iteration_count", 0) >= 3:
        sys.stderr.write("[Phase 5] Max DevOps validation retries (3) reached.\n")
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


# =====================================================================
# 5. MCP TOOL REGISTRATION
# =====================================================================

@mcp.tool()
async def generate_devops_pipeline(
    deploy_id: str = "",
    workspace_path: str = ".",
) -> str:
    """
    Executes Phase 5 of SDLC: Generates multi-stage non-root Dockerfile, docker-compose.yml,
    and .github/workflows/ci.yml, validating YAML and Compose schemas before completion.
    """
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
        if AsyncSqliteSaver is not None:
            os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
            db_path = root / ".orchestrator" / "checkpoints.sqlite"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                graph = build_devops_graph(checkpointer=saver)
                final_state = await graph.ainvoke(initial_state, config=config)
        else:
            graph = build_devops_graph(checkpointer=get_sqlite_checkpointer(root))
            final_state = await asyncio.to_thread(graph.invoke, initial_state, config)

        passed = final_state.get("validation_exit_code", 1) == 0
        iters = final_state.get("iteration_count", 0)
        status = "DEVOPS_VALIDATED" if passed else "DEVOPS_GENERATED_WITH_WARNINGS"

        log_audit_event(
            workspace=root,
            phase="phase5_devops",
            thread_id=thread_id,
            status=status,
            iterations=iters,
        )

        return (
            f"Phase 5 DevOps {status} in {iters} pass(es). "
            f"Generated Dockerfile, docker-compose.yml, and .github/workflows/ci.yml."
        )


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "http":
        mcp.run(
            transport="streamable-http",
            host=os.getenv("MCP_HOST", "127.0.0.1"),
            port=int(os.getenv("MCP_PORT", "8005")),
        )
    else:
        mcp.run()