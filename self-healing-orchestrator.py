import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
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

mcp = FastMCP("SDLC-Observability-SelfHealing-Orchestrator")


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


def sre_triage_agent(state: SelfHealingState):
    sys.stderr.write("\n[Phase 6 -> Agent 1] SRE Triage Agent analyzing runtime telemetry & stack trace...\n")
    workspace = Path(state["workspace_path"])
    llm = get_llm()
    system_spec = load_agent_prompt("sre_triage", ["sre_incident_contracts"], workspace=workspace)

    prompt = f"""{system_spec}

[Production Incident Logs / Alert Payload]
{state['incident_logs']}

[Application Source Code]
{state['codebase_snapshot']}

[Architecture Context]
{state['arch_text'][:2500]}"""

    diagnosis = llm.with_structured_output(IncidentDiagnosis).invoke(prompt)
    sys.stderr.write(f"[Phase 6] Triaged: [{diagnosis.severity}] {diagnosis.incident_title}\n")
    return {"diagnosis": diagnosis}


def hotfix_and_regression_engineer(state: SelfHealingState):
    sys.stderr.write("\n[Phase 6 -> Agent 2] Generating regression test & runtime hotfix...\n")
    diag = state["diagnosis"]
    workspace = Path(state["workspace_path"])
    llm = get_llm()
    is_retry = state.get("iteration_count", 0) > 0 and state.get("last_test_exit_code", 0) != 0

    system_spec = load_agent_prompt(
        "hotfix_engineer",
        ["python_validation_contracts", "sre_incident_contracts", "markdown_file_output"],
        workspace=workspace,
        include_retry_skill=is_retry,
    )
    existing_tests_ctx = get_existing_tests(workspace)

    retry_context = ""
    if is_retry:
        last_written = "\n\n".join(f"### {f.file_path}\n{f.content}" for f in state.get("written_artifacts", []))
        retry_context = (
            f"\n[PREVIOUS HOTFIX VERIFICATION FAILED (Exit Code {state['last_test_exit_code']})]\n"
            f"Stdout: {state['last_test_stdout'][-1500:]}\n"
            f"Stderr: {state['last_test_stderr'][-1500:]}\n"
            f"Files written on last attempt:\n{last_written[:2000]}\n"
        )

    prompt = f"""{system_spec}

[Diagnosed Incident]
- Incident: {diag.incident_title} ({diag.severity})
- Root Cause: {diag.root_cause_analysis}
- Reproduction Steps: {diag.reproduction_steps}
- Initial Fix Strategy: {diag.fix_strategy}
- Affected Files: {diag.affected_files}
{retry_context}
[Current Source Codebase]
{state['codebase_snapshot']}

[Existing Unit & Integration Tests (MUST CONTINUE TO PASS)]
{existing_tests_ctx}"""

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
        return {"last_test_stdout": "", "last_test_stderr": syntax_err, "last_test_exit_code": 2, "iteration_count": new_iter}

    result = run_sandboxed_tests(workspace, command=state.get("test_runner_command") or None, timeout_seconds=45)
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
    llm = get_llm()
    passed = state.get("last_test_exit_code", 1) == 0

    alerts_spec = load_agent_prompt("observability_architect", ["markdown_file_output"], workspace=workspace)
    alerts_prompt = f"""{alerts_spec}

Generate Prometheus alerting rules (`monitoring/alerts.yml`) for:
- Incident: {diag.incident_title} ({diag.severity})
- Root Cause: {diag.root_cause_analysis}"""
    alerts_res = llm.invoke(alerts_prompt)
    extract_and_write_files(workspace, alerts_res.content)

    pm_spec = load_agent_prompt("observability_architect", ["sre_incident_contracts"], workspace=workspace)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    postmortem_prompt = f"""{pm_spec}

Write `POSTMORTEM.md` covering:
- Incident: {diag.incident_title} ({diag.severity})
- Root Cause Analysis (5 Whys): {diag.root_cause_analysis}
- Verification & Regression Test Coverage (Tests Passed: {passed})
- Action Items & Monitoring Safeguards Added

Sign off with:
**Prepared by:** Autonomous SRE Orchestrator (Owner: Gaddam Rohith Reddy)
**Date:** {timestamp}

Output ONLY clean Markdown."""

    pm_res = llm.invoke(postmortem_prompt)
    docs_dir = workspace / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "POSTMORTEM.md").write_text(pm_res.content, encoding="utf-8")
    return state


def route_hotfix_verification(state: SelfHealingState):
    if state.get("last_test_exit_code", 1) == 0 or state.get("iteration_count", 0) >= 3:
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


@mcp.tool()
async def heal_runtime_incident(
    incident_logs: str,
    incident_id: str = "inc-auto",
    test_runner_command: str = "",
    workspace_path: str = ".",
) -> str:
    """Executes Phase 6 of SDLC using dynamic agents.md & skills.md specs inside an isolated worktree."""
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
            try:
                if AsyncSqliteSaver is not None:
                    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
                    db_path = root / ".orchestrator" / "checkpoints.sqlite"
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                        final_state = await build_self_healing_graph(checkpointer=saver).ainvoke(initial_state, config=config)
                else:
                    final_state = await asyncio.to_thread(
                        build_self_healing_graph(checkpointer=get_sqlite_checkpointer(root)).invoke, initial_state, config
                    )
            except TokenBudgetExceededError as e:
                log_audit_event(root, "phase6_self_healing", thread_id, "TOKEN_BUDGET_HALTED", 0, {"error": str(e)})
                return f"Phase 6 HALTED by token circuit breaker: {e}"

            diag = final_state.get("diagnosis")
            passed = final_state.get("last_test_exit_code", 1) == 0
            iters = final_state.get("iteration_count", 0)
            status = "RESOLVED_VERIFIED" if passed else "PATCHED_MANUAL_REVIEW"

            log_audit_event(
                root,
                "phase6_self_healing",
                thread_id,
                status,
                iters,
                {"severity": diag.severity if diag else "UNKNOWN", "sandbox": str(sandbox_dir)},
            )
            return (
                f"Phase 6 Self-Healing {status} for [{diag.severity if diag else 'INCIDENT'}] "
                f"{diag.incident_title if diag else ''}. Isolated worktree: {sandbox_dir}"
            )


if __name__ == "__main__":
    if os.getenv("MCP_TRANSPORT", "stdio").lower() == "http":
        mcp.run(transport="streamable-http", host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8006")))
    else:
        mcp.run()