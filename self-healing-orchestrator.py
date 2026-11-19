import asyncio
import os
import sys
from datetime import datetime
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

mcp = FastMCP("SDLC-Observability-SelfHealing-Orchestrator")

llm = get_llm()
#llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0)
# llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.0, num_ctx=4096)


# =====================================================================
# 2. STRUCTURED SCHEMAS & STATE
# =====================================================================

class IncidentDiagnosis(BaseModel):
    incident_title: str = Field(description="Concise SRE incident title")
    severity: str = Field(description="'SEV-1', 'SEV-2', or 'SEV-3'")
    affected_files: List[str] = Field(description="Relative paths of source files causing the failure")
    root_cause_analysis: str = Field(description="Deep technical explanation of why the runtime failure occurred")
    reproduction_steps: str = Field(description="Exact inputs/state required to reproduce the bug in a unit test")
    fix_strategy: str = Field(description="Code-level remediation plan")


class SelfHealingState(TypedDict):
    workspace_path: str
    incident_logs: str
    test_runner_command: str
    codebase_snapshot: str
    arch_text: str

    diagnosis: Optional[IncidentDiagnosis]
    written_artifacts: List[ArtifactFile]
    pr_summary: str

    last_test_stdout: str
    last_test_stderr: str
    last_test_exit_code: int
    iteration_count: int


# =====================================================================
# 3. AGENT NODES
# =====================================================================

def sre_triage_agent(state: SelfHealingState):
    sys.stderr.write("\n[Phase 6 -> Agent 1] SRE Triage Agent analyzing runtime telemetry & stack trace...\n")
    prompt = f"""
    You are a Principal Site Reliability Engineer (SRE) responding to a production incident.
    Correlate the runtime error logs / telemetry alert below with the application source code.

    [Production Incident Logs / Alert Payload]
    {state['incident_logs']}

    [Application Source Code]
    {state['codebase_snapshot']}

    [Architecture Context]
    {state['arch_text'][:2500]}

    Tasks:
    1. Classify incident severity ('SEV-1', 'SEV-2', or 'SEV-3').
    2. Identify the exact source file(s) and function(s) responsible.
    3. Determine the root cause.
    4. Define exact reproduction inputs so an engineer can write a failing regression test.
    """
    diagnosis = llm.with_structured_output(IncidentDiagnosis).invoke(prompt)
    sys.stderr.write(f"[Phase 6] Triaged: [{diagnosis.severity}] {diagnosis.incident_title}\n")
    return {"diagnosis": diagnosis}


def hotfix_and_regression_engineer(state: SelfHealingState):
    sys.stderr.write("\n[Phase 6 -> Agent 2] Generating regression test & runtime hotfix...\n")
    diag = state["diagnosis"]
    workspace = Path(state["workspace_path"])
    existing_tests_ctx = get_existing_tests(workspace)

    retry_context = ""
    if state.get("iteration_count", 0) > 0 and state.get("last_test_exit_code", 0) != 0:
        last_written = "\n\n".join(f"### {f.file_path}\n{f.content}" for f in state.get("written_artifacts", []))
        retry_context = (
            f"\nPREVIOUS HOTFIX VERIFICATION FAILED (Exit Code {state['last_test_exit_code']}):\n"
            f"Stdout: {state['last_test_stdout'][-1500:]}\n"
            f"Stderr: {state['last_test_stderr'][-1500:]}\n"
            f"Files written on last attempt:\n{last_written[:2000]}\n"
            f"IMPORTANT: Adjust your implementation and regression test so BOTH the existing test suite "
            f"and the new regression test pass together.\n"
        )

    fence = "`" * 3
    prompt = f"""
    You are a Staff Sustaining & Hotfix Software Engineer.
    A production incident was diagnosed:

    - Incident: {diag.incident_title} ({diag.severity})
    - Root Cause: {diag.root_cause_analysis}
    - Reproduction Steps: {diag.reproduction_steps}
    - Initial Fix Strategy: {diag.fix_strategy}
    - Affected Files: {diag.affected_files}
    {retry_context}

    [Current Source Codebase]
    {state['codebase_snapshot']}

    [Existing Unit & Integration Tests (MUST CONTINUE TO PASS)]
    {existing_tests_ctx}

    Mandatory Deliverables & Contracts:
    1. `tests/regression/test_incident_regression.py`: Automated pytest regression test reproducing the crash 
       condition and asserting the fixed behavior. Import modules directly from `src/`.
       Assert exception types (e.g., `pytest.raises(TypeError)` for `None`/invalid types and `pytest.raises(ValueError)` 
       for out-of-range values) rather than brittle exact exception message strings.
    2. Patched source file(s) under `src/` that fix the crash while preserving backward compatibility 
       with all existing tests above:
       - Keep type checks (`TypeError` for `None`/bool/non-numeric) and range checks (`ValueError` for `<= 0`) separate.
       - Preserve exact exception types AND exact error message strings asserted by existing tests in `[Existing Unit & Integration Tests]`.
       - Accept valid `int` and `float` values `> 0`.

    Output EVERY file strictly using this Markdown format (do NOT output JSON):

    FILE: tests/regression/test_incident_regression.py
    {fence}python
    # complete multi-line regression test here
    {fence}

    FILE: src/example.py
    {fence}python
    # complete multi-line patched source code here
    {fence}
    """
    response = llm.invoke(prompt)
    written = extract_and_write_files(workspace, response.content)
    return {
        "written_artifacts": written,
        "codebase_snapshot": get_source_snapshot(workspace),
        "pr_summary": f"Hotfix [{diag.severity}]: {diag.incident_title} - {diag.fix_strategy}",
    }


def regression_verification_runner(state: SelfHealingState):
    workspace = Path(state["workspace_path"])
    new_iter = state.get("iteration_count", 0) + 1
    sys.stderr.write(f"\n[Phase 6 -> Runner] Executing verification Pass #{new_iter}...\n")

    syntax_err = validate_python_ast(state.get("written_artifacts", []))
    if syntax_err:
        sys.stderr.write(f"[Phase 6 -> AST Gate] {syntax_err}\n")
        return {
            "last_test_stdout": "",
            "last_test_stderr": syntax_err,
            "last_test_exit_code": 2,
            "iteration_count": new_iter,
        }

    cmd = state.get("test_runner_command") or None
    result = run_sandboxed_tests(workspace, command=cmd, timeout_seconds=45)
    sys.stderr.write(f"[Phase 6 -> Runner] Pass #{new_iter} Exit Code: {result.exit_code}\n")
    return {
        "last_test_stdout": result.stdout,
        "last_test_stderr": result.stderr,
        "last_test_exit_code": result.exit_code,
        "iteration_count": new_iter,
    }


def observability_and_postmortem_agent(state: SelfHealingState):
    sys.stderr.write("\n[Phase 6 -> Agent 3] Hardening observability alerts & generating Post-Mortem...\n")
    diag = state["diagnosis"]
    workspace = Path(state["workspace_path"])
    passed = state.get("last_test_exit_code", 1) == 0
    fence = "`" * 3

    alerts_prompt = f"""
    You are a Principal Observability & Site Reliability Architect.
    Generate Prometheus alerting rules (`monitoring/alerts.yml`) to proactively detect this failure mode:
    Incident: {diag.incident_title} ({diag.severity})
    Root Cause: {diag.root_cause_analysis}

    Output strictly using this Markdown format:

    FILE: monitoring/alerts.yml
    {fence}yaml
    # Prometheus alert rules here
    {fence}
    """
    alerts_res = llm.invoke(alerts_prompt)
    extract_and_write_files(workspace, alerts_res.content)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    postmortem_prompt = f"""
    Write a comprehensive, blameless SRE Incident Post-Mortem (`POSTMORTEM.md`) covering:
    - Executive Summary & Impact ({diag.incident_title}, {diag.severity})
    - Root Cause Analysis (5 Whys): {diag.root_cause_analysis}
    - Resolution & Regression Test Coverage (Tests Passed: {passed})
    - Action Items & Monitoring Safeguards Added

    Sign off the document with:
    **Prepared by:** Autonomous SRE Orchestrator (Owner: Gaddam Rohith Reddy)
    **Date:** {timestamp}

    Output ONLY clean Markdown with zero bracketed placeholders.
    """
    pm_res = llm.invoke(postmortem_prompt)
    docs_dir = workspace / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "POSTMORTEM.md").write_text(pm_res.content, encoding="utf-8")
    sys.stderr.write(f"[Phase 6] Post-mortem saved to: {(docs_dir / 'POSTMORTEM.md').resolve()}\n")
    return state


# =====================================================================
# 4. GRAPH COMPILATION HELPER
# =====================================================================

def route_hotfix_verification(state: SelfHealingState):
    if state.get("last_test_exit_code", 1) == 0:
        return "verified"
    if state.get("iteration_count", 0) >= 3:
        sys.stderr.write("[Phase 6] Max self-healing repair attempts (3) reached.\n")
        return "verified"
    return "retry_hotfix"


def build_self_healing_graph(checkpointer=None):
    workflow = StateGraph(SelfHealingState)
    workflow.add_node("sre_triage", sre_triage_agent)
    workflow.add_node("hotfix_engineer", hotfix_and_regression_engineer)
    workflow.add_node("verifier", regression_verification_runner)
    workflow.add_node("postmortem_writer", observability_and_postmortem_agent)

    workflow.add_edge(START, "sre_triage")
    workflow.add_edge("sre_triage", "hotfix_engineer")
    workflow.add_edge("hotfix_engineer", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        route_hotfix_verification,
        {"verified": "postmortem_writer", "retry_hotfix": "hotfix_engineer"},
    )
    workflow.add_edge("postmortem_writer", END)
    return workflow.compile(checkpointer=checkpointer)


# =====================================================================
# 5. MCP TOOL REGISTRATION
# =====================================================================

@mcp.tool()
async def heal_runtime_incident(
    incident_logs: str,
    incident_id: str = "inc-auto",
    test_runner_command: str = "",
    workspace_path: str = ".",
) -> str:
    """
    Executes Phase 6 of SDLC inside an isolated Git worktree with SQLite checkpointing,
    AST validation, scrubbed-environment pytest verification, Prometheus alerts, and POSTMORTEM.md.
    """
    root = Path(workspace_path).resolve()
    thread_id = f"phase6-{incident_id.lower()}"

    async with get_workspace_lock(root):
        with git_worktree_sandbox(root, f"hotfix-{incident_id}") as sandbox_dir:
            arch_file = sandbox_dir / "docs" / "architecture.md"
            initial_state: SelfHealingState = {
                "workspace_path": str(sandbox_dir),
                "incident_logs": incident_logs.strip(),
                "test_runner_command": test_runner_command.strip(),
                "codebase_snapshot": get_source_snapshot(sandbox_dir),
                "arch_text": arch_file.read_text(encoding="utf-8") if arch_file.exists() else "",
                "diagnosis": None,
                "written_artifacts": [],
                "pr_summary": "",
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
                    graph = build_self_healing_graph(checkpointer=saver)
                    final_state = await graph.ainvoke(initial_state, config=config)
            else:
                graph = build_self_healing_graph(checkpointer=get_sqlite_checkpointer(root))
                final_state = await asyncio.to_thread(graph.invoke, initial_state, config)

            diag = final_state.get("diagnosis")
            passed = final_state.get("last_test_exit_code", 1) == 0
            iters = final_state.get("iteration_count", 0)
            status = "RESOLVED_VERIFIED" if passed else "PATCHED_MANUAL_REVIEW"

            log_audit_event(
                workspace=root,
                phase="phase6_self_healing",
                thread_id=thread_id,
                status=status,
                iterations=iters,
                details={
                    "severity": diag.severity if diag else "UNKNOWN",
                    "title": diag.incident_title if diag else "",
                    "sandbox": str(sandbox_dir),
                },
            )

            return (
                f"Phase 6 Self-Healing {status} for [{diag.severity if diag else 'INCIDENT'}] "
                f"{diag.incident_title if diag else ''}. "
                f"Isolated worktree: {sandbox_dir} | Post-mortem: {sandbox_dir}/docs/POSTMORTEM.md"
            )


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "http":
        mcp.run(
            transport="streamable-http",
            host=os.getenv("MCP_HOST", "127.0.0.1"),
            port=int(os.getenv("MCP_PORT", "8006")),
        )
    else:
        mcp.run()