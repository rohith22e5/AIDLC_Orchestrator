import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from fastmcp import FastMCP

from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:
    AsyncSqliteSaver = None

from orchestrator_core import (
    ArtifactFile,
    extract_and_write_files,
    get_existing_tests,
    get_source_snapshot,
    get_sqlite_checkpointer,
    get_workspace_lock,
    git_worktree_sandbox,
    log_audit_event,
    run_sandboxed_tests,
    validate_python_ast,
    get_llm

)

# =====================================================================
# 1. INITIALIZE FASTMCP SERVER & MODEL
# =====================================================================

mcp = FastMCP("SDLC-Coding-TDD-Orchestrator")

llm = get_llm()
#llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0)
# llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.0, num_ctx=4096)


# =====================================================================
# 2. STATE & SCHEMAS
# =====================================================================

class StoryTask(BaseModel):
    story_id: str = Field(description="User story identifier e.g., US-01")
    title: str = Field(description="Concise story summary")
    acceptance_criteria: List[str] = Field(description="Testable conditions required for completion")

class CodingState(TypedDict):
    workspace_path: str
    story_id: str
    story_spec: str
    arch_text: str
    plan_text: str
    test_runner_command: str

    codebase_snapshot: str
    written_files: List[ArtifactFile]

    last_test_stdout: str
    last_test_stderr: str
    last_test_exit_code: int
    iteration_count: int


# =====================================================================
# 3. AGENT NODES
# =====================================================================

def test_engineer_agent(state: CodingState):
    sys.stderr.write(f"\n[Phase 2 -> Agent 1] Test Engineer writing pytest suite for {state['story_id']}...\n")
    workspace = Path(state["workspace_path"])
    existing_tests = get_existing_tests(workspace)
    fence = "`" * 3

    prompt = f"""
    You are a Principal QA & TDD Automation Engineer.
    Write comprehensive pytest unit tests BEFORE implementation for the story below.

    [User Story Spec]
    {state['story_spec']}

    [Architecture & Plan Context]
    {state['arch_text'][:2500]}
    {state['plan_text'][:2000]}

    [Current Source Code]
    {state['codebase_snapshot']}

    [Existing Tests (Preserve compatibility)]
    {existing_tests}

    Rules:
    - Write tests under `tests/test_solution.py` (or story-specific `tests/test_{state['story_id'].lower().replace('-', '_')}.py`).
    - Import modules directly from `src/` (e.g., `from cache import TTLCache`).
    - Assert `TypeError` for invalid types (`None`, `bool`, non-numeric) and `ValueError` for invalid value ranges (`<= 0`).
    - Keep any sleep durations fast (< 0.2s).

    Output strictly using this Markdown format (do NOT output JSON):

    FILE: tests/test_solution.py
    {fence}python
    # complete multi-line pytest file here
    {fence}
    """
    res = llm.invoke(prompt)
    written = extract_and_write_files(workspace, res.content)
    return {"written_files": written}


def software_engineer_agent(state: CodingState):
    sys.stderr.write(f"\n[Phase 2 -> Agent 2] Software Engineer implementing {state['story_id']}...\n")
    workspace = Path(state["workspace_path"])
    existing_tests = get_existing_tests(workspace, exclude_dirs=())
    fence = "`" * 3

    retry_note = ""
    if state.get("iteration_count", 0) > 0 and state.get("last_test_exit_code", 0) != 0:
        last_written = "\n\n".join(f"### {f.file_path}\n{f.content}" for f in state.get("written_files", []))
        retry_note = (
            f"\nPREVIOUS TEST RUN FAILED (Exit Code {state['last_test_exit_code']}):\n"
            f"Stdout: {state['last_test_stdout'][-1500:]}\n"
            f"Stderr: {state['last_test_stderr'][-1500:]}\n"
            f"Previously written files:\n{last_written[:2000]}\n"
            f"Fix all syntax, import, type, and assertion failures.\n"
        )

    prompt = f"""
    You are a Staff Backend Software Engineer implementing a user story via strict TDD.

    [User Story Spec]
    {state['story_spec']}

    [Architecture Context]
    {state['arch_text'][:2500]}
    {retry_note}

    [Current Source Codebase]
    {state['codebase_snapshot']}

    [Target Test Suite (MUST PASS)]
    {existing_tests}

    Rules:
    - Implement clean, production-grade Python code under `src/`.
    - Use `threading.RLock()` (never `threading.Lock()`) if locking is needed.
    - Keep type validation (`TypeError`) and value-range validation (`ValueError`) in separate `if` checks, and preserve exact error strings expected by existing tests.

    Output EVERY modified/created file strictly using this Markdown format:

    FILE: src/solution.py
    {fence}python
    # complete multi-line implementation here
    {fence}
    """
    res = llm.invoke(prompt)
    written = extract_and_write_files(workspace, res.content)
    return {
        "written_files": written,
        "codebase_snapshot": get_source_snapshot(workspace),
    }


def tdd_verification_runner(state: CodingState):
    workspace = Path(state["workspace_path"])
    new_iter = state.get("iteration_count", 0) + 1
    sys.stderr.write(f"\n[Phase 2 -> Runner] Verifying Pass #{new_iter}...\n")

    syntax_err = validate_python_ast(state.get("written_files", []))
    if syntax_err:
        sys.stderr.write(f"[Phase 2 -> AST Gate] {syntax_err}\n")
        return {
            "last_test_stdout": "",
            "last_test_stderr": syntax_err,
            "last_test_exit_code": 2,
            "iteration_count": new_iter,
        }

    cmd = state.get("test_runner_command") or None
    result = run_sandboxed_tests(workspace, command=cmd, timeout_seconds=30)
    sys.stderr.write(f"[Phase 2 -> Runner] Exit Code: {result.exit_code}\n")
    return {
        "last_test_stdout": result.stdout,
        "last_test_stderr": result.stderr,
        "last_test_exit_code": result.exit_code,
        "iteration_count": new_iter,
    }


# =====================================================================
# 4. GRAPH COMPILATION HELPER
# =====================================================================

def route_tdd_loop(state: CodingState):
    if state.get("last_test_exit_code", 1) == 0:
        return "done"
    if state.get("iteration_count", 0) >= 3:
        sys.stderr.write("[Phase 2] Max TDD repair iterations (3) reached.\n")
        return "done"
    return "retry"


def build_coding_graph(checkpointer=None):
    builder = StateGraph(CodingState)
    builder.add_node("test_engineer", test_engineer_agent)
    builder.add_node("software_engineer", software_engineer_agent)
    builder.add_node("verifier", tdd_verification_runner)

    builder.add_edge(START, "test_engineer")
    builder.add_edge("test_engineer", "software_engineer")
    builder.add_edge("software_engineer", "verifier")
    builder.add_conditional_edges("verifier", route_tdd_loop, {"done": END, "retry": "software_engineer"})
    return builder.compile(checkpointer=checkpointer)


# =====================================================================
# 5. MCP TOOL REGISTRATION
# =====================================================================

@mcp.tool()
async def implement_and_verify_story(
    story_id: str,
    story_spec: str,
    test_runner_command: str = "",
    workspace_path: str = ".",
) -> str:
    """
    Executes Phase 2 (TDD Coding & Verification) inside an isolated Git worktree with SQLite
    checkpointing, AST syntax gates, and scrubbed-environment pytest verification.
    """
    root = Path(workspace_path).resolve()
    thread_id = f"phase2-{story_id.lower()}"

    async with get_workspace_lock(root):
        with git_worktree_sandbox(root, f"phase2-{story_id}") as sandbox_dir:
            arch_file = sandbox_dir / "docs" / "architecture.md"
            plan_file = sandbox_dir / "docs" / "plan.md"

            initial_state: CodingState = {
                "workspace_path": str(sandbox_dir),
                "story_id": story_id,
                "story_spec": story_spec.strip(),
                "arch_text": arch_file.read_text(encoding="utf-8") if arch_file.exists() else "",
                "plan_text": plan_file.read_text(encoding="utf-8") if plan_file.exists() else "",
                "test_runner_command": test_runner_command.strip(),
                "codebase_snapshot": get_source_snapshot(sandbox_dir),
                "written_files": [],
                "last_test_stdout": "",
                "last_test_stderr": "",
                "last_test_exit_code": 1,
                "iteration_count": 0,
            }

            config = {"configurable": {"thread_id": thread_id}}
            if AsyncSqliteSaver is not None:
                os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
                db_path = root / ".orchestrator" / "checkpoints.sqlite"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                    graph = build_coding_graph(checkpointer=saver)
                    final_state = await graph.ainvoke(initial_state, config=config)
            else:
                graph = build_coding_graph(checkpointer=get_sqlite_checkpointer(root))
                final_state = await asyncio.to_thread(graph.invoke, initial_state, config)

            passed = final_state.get("last_test_exit_code", 1) == 0
            iters = final_state.get("iteration_count", 0)
            status = "VERIFIED" if passed else "FAILED_MAX_RETRIES"

            log_audit_event(
                workspace=root,
                phase="phase2_coding",
                thread_id=thread_id,
                status=status,
                iterations=iters,
                details={"story_id": story_id, "sandbox": str(sandbox_dir)},
            )

            return (
                f"Phase 2 TDD {status} for [{story_id}] in {iters} pass(es). "
                f"Isolated worktree: {sandbox_dir}"
            )


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "http":
        mcp.run(
            transport="streamable-http",
            host=os.getenv("MCP_HOST", "127.0.0.1"),
            port=int(os.getenv("MCP_PORT", "8002")),
        )
    else:
        mcp.run()