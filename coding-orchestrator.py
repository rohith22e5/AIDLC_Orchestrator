import asyncio
import os
import sys
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
    extract_and_write_files,
    get_existing_tests,
    get_llm,
    get_source_snapshot,
    get_sqlite_checkpointer,
    get_workspace_lock,
    git_worktree_sandbox,
    load_agent_prompt,
    log_audit_event,
    run_sandboxed_tests,
    validate_python_ast,
)

mcp = FastMCP("SDLC-Coding-TDD-Orchestrator")


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


def test_engineer_agent(state: CodingState):
    sys.stderr.write(f"\n[Phase 2 -> Agent 1] Test Engineer writing pytest suite for {state['story_id']}...\n")
    workspace = Path(state["workspace_path"])
    llm = get_llm("test_engineer")
    system_spec = load_agent_prompt(
        "test_engineer",
        ["python_validation_contracts", "markdown_file_output"],
        workspace=workspace,
    )
    existing_tests = get_existing_tests(workspace)

    prompt = f"""{system_spec}

[User Story Spec: {state['story_id']}]
{state['story_spec']}

[Architecture & Plan Context]
{state['arch_text'][:2500]}
{state['plan_text'][:2000]}

[Current Source Code]
{state['codebase_snapshot']}

[Existing Tests (Preserve compatibility)]
{existing_tests}"""

    res = llm.invoke(prompt)
    written = extract_and_write_files(workspace, res.content, allowed_prefixes=("tests/",))
    return {"written_files": written}


def test_critic_agent(state: CodingState):
    """Independent Granite 4.2 auditor verifying test math & imports before implementation."""
    sys.stderr.write(f"\n[Phase 2 -> Gate] Test Critic auditing generated pytest suite for {state['story_id']}...\n")
    workspace = Path(state["workspace_path"])
    llm = get_llm("appsec_reviewer")
    system_spec = load_agent_prompt(
        "test_critic",
        ["python_validation_contracts", "markdown_file_output"],
        workspace=workspace,
    )
    current_tests = get_existing_tests(workspace, exclude_dirs=(), max_chars=5000)

    failure_ctx = ""
    if state.get("iteration_count", 0) > 0 and state.get("last_test_exit_code", 0) != 0:
        failure_ctx = (
            f"\n[Implementation failed against these tests twice. Check if a test assertion contradicts the spec:]\n"
            f"Stdout: {state['last_test_stdout'][-1200:]}\n"
            f"Stderr: {state['last_test_stderr'][-1200:]}\n"
        )

    prompt = f"""{system_spec}

[User Story Spec: {state['story_id']}]
{state['story_spec']}

[Architecture Contract]
{state['arch_text'][:2500]}
{failure_ctx}
[Current Test Suite Under Audit]
{current_tests}"""

    res = llm.invoke(prompt)
    if "TESTS_APPROVED" not in res.content:
        extract_and_write_files(workspace, res.content, allowed_prefixes=("tests/",))
    return {}


def software_engineer_agent(state: CodingState):
    sys.stderr.write(f"\n[Phase 2 -> Agent 2] Software Engineer implementing {state['story_id']}...\n")
    workspace = Path(state["workspace_path"])
    iters = state.get("iteration_count", 0)
    is_retry = iters > 0 and state.get("last_test_exit_code", 0) != 0
    llm = get_llm("software_engineer", escalate=(iters >= 2))

    system_spec = load_agent_prompt(
        "software_engineer",
        ["python_validation_contracts", "markdown_file_output"],
        workspace=workspace,
        include_retry_skill=is_retry,
    )
    existing_tests = get_existing_tests(workspace, exclude_dirs=())

    retry_note = ""
    if is_retry:
        last_written = "\n\n".join(f"### {f.file_path}\n{f.content}" for f in state.get("written_files", []))
        retry_note = (
            f"\n[PREVIOUS TEST RUN FAILED (Exit Code {state['last_test_exit_code']})]\n"
            f"Stdout: {state['last_test_stdout'][-1500:]}\n"
            f"Stderr: {state['last_test_stderr'][-1500:]}\n"
            f"Previously written files:\n{last_written[:2000]}\n"
        )

    prompt = f"""{system_spec}

[User Story Spec: {state['story_id']}]
{state['story_spec']}

[Architecture Context]
{state['arch_text'][:2500]}
{retry_note}
[Current Source Codebase]
{state['codebase_snapshot']}

[Immutable Test Suite (MUST PASS — DO NOT EDIT TESTS)]
{existing_tests}"""

    res = llm.invoke(prompt)
    written = extract_and_write_files(workspace, res.content, allowed_prefixes=("src/",))
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
        return {"last_test_stdout": "", "last_test_stderr": syntax_err, "last_test_exit_code": 2, "iteration_count": new_iter}

    result = run_sandboxed_tests(workspace, command=state.get("test_runner_command") or None, timeout_seconds=30)
    sys.stderr.write(f"[Phase 2 -> Runner] Exit Code: {result.exit_code}\n")
    return {
        "last_test_stdout": result.stdout,
        "last_test_stderr": result.stderr,
        "last_test_exit_code": result.exit_code,
        "iteration_count": new_iter,
    }


def route_tdd_loop(state: CodingState):
    if state.get("last_test_exit_code", 1) == 0 or state.get("iteration_count", 0) >= 3:
        return "done"
    if state.get("iteration_count", 0) == 2:
        return "audit_tests"
    return "retry"


def build_coding_graph(checkpointer=None):
    builder = StateGraph(CodingState)
    builder.add_node("test_engineer", test_engineer_agent)
    builder.add_node("test_critic", test_critic_agent)
    builder.add_node("software_engineer", software_engineer_agent)
    builder.add_node("verifier", tdd_verification_runner)

    builder.add_edge(START, "test_engineer")
    builder.add_edge("test_engineer", "test_critic")
    builder.add_edge("test_critic", "software_engineer")
    builder.add_edge("software_engineer", "verifier")
    builder.add_conditional_edges(
        "verifier",
        route_tdd_loop,
        {"done": END, "retry": "software_engineer", "audit_tests": "test_critic"},
    )
    return builder.compile(checkpointer=checkpointer)


@mcp.tool()
async def implement_and_verify_story(
    story_id: str,
    story_spec: str,
    test_runner_command: str = "",
    workspace_path: str = ".",
) -> str:
    """Executes Phase 2 (TDD Coding & Verification) with role-isolated write locks."""
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
            try:
                if AsyncSqliteSaver is not None:
                    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
                    db_path = root / ".orchestrator" / "checkpoints.sqlite"
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                        final_state = await build_coding_graph(checkpointer=saver).ainvoke(initial_state, config=config)
                else:
                    final_state = await asyncio.to_thread(
                        build_coding_graph(checkpointer=get_sqlite_checkpointer(root)).invoke, initial_state, config
                    )
            except TokenBudgetExceededError as e:
                log_audit_event(root, "phase2_coding", thread_id, "TOKEN_BUDGET_HALTED", 0, {"error": str(e)})
                return f"Phase 2 HALTED by token circuit breaker: {e}"

            passed = final_state.get("last_test_exit_code", 1) == 0
            iters = final_state.get("iteration_count", 0)
            status = "VERIFIED" if passed else "FAILED_MAX_RETRIES"

            log_audit_event(root, "phase2_coding", thread_id, status, iters, {"story_id": story_id, "sandbox": str(sandbox_dir)})
            return f"Phase 2 TDD {status} for [{story_id}] in {iters} pass(es). Isolated worktree: {sandbox_dir}"


if __name__ == "__main__":
    if os.getenv("MCP_TRANSPORT", "stdio").lower() == "http":
        mcp.run(transport="streamable-http", host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8002")))
    else:
        mcp.run()