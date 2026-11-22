import asyncio
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from fastmcp import FastMCP

from langgraph.graph import StateGraph, START, END

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:
    AsyncSqliteSaver = None

from orchestrator_core import (
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

mcp = FastMCP("SDLC-AppSec-CodeReview-Orchestrator")


class SecurityAuditReport(BaseModel):
    risk_level: str = Field(description="'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'")
    approved_for_merge: bool = Field(description="True if no HIGH or CRITICAL issues exist")
    findings: List[str] = Field(description="Specific security, concurrency, or code quality findings")
    remediation_summary: str = Field(description="Concrete fixes required if not approved")


class ReviewState(TypedDict):
    workspace_path: str
    codebase_snapshot: str
    arch_text: str
    static_scan_output: str
    audit_report: Optional[SecurityAuditReport]


def _run_single_scanner(tool: str, args: List[str], workspace: Path, env: dict) -> str:
    exe = shutil.which(tool)
    if not exe:
        return f"[{tool.upper()}] not installed in PATH (skipped)"
    try:
        proc = subprocess.run([exe, *args], cwd=workspace, env=env, capture_output=True, text=True, timeout=30)
        return f"[{tool.upper()} exit={proc.returncode}]\n{(proc.stdout or proc.stderr)[:1500]}"
    except Exception as e:
        return f"[{tool.upper()} error] {e}"


async def static_security_scanner_node(state: ReviewState):
    sys.stderr.write("\n[Phase 3 -> Step 1] Running concurrent static & security scanners...\n")
    workspace = Path(state["workspace_path"])
    env = build_scrubbed_env(workspace)
    scan_target = "src" if (workspace / "src").exists() else "."

    scanners: List[Tuple[str, List[str]]] = [
        ("ruff", ["check", scan_target]),
        ("bandit", ["-r", scan_target, "-q"]),
        ("pip-audit", ["--progress-spinner", "off"]),
    ]
    results = await asyncio.gather(
        *(asyncio.to_thread(_run_single_scanner, tool, args, workspace, env) for tool, args in scanners)
    )
    return {"static_scan_output": "\n\n".join(results)}


def appsec_and_quality_reviewer_agent(state: ReviewState):
    sys.stderr.write("\n[Phase 3 -> Step 2] AppSec & Staff Reviewer auditing codebase...\n")
    workspace = Path(state["workspace_path"])
    llm = get_llm()
    system_spec = load_agent_prompt("appsec_reviewer", ["python_validation_contracts"], workspace=workspace)
    fence = "`" * 3

    prompt = f"""{system_spec}

[Architecture Contract]
{state['arch_text'][:2500]}

[Deterministic Static Scanner Results]
{state['static_scan_output']}

[Application Source Code]
{state['codebase_snapshot']}"""

    report = llm.with_structured_output(SecurityAuditReport).invoke(prompt)

    md_lines = [
        "# Security & Code Review Report",
        f"- **Risk Level:** {report.risk_level}",
        f"- **Approved for Merge:** {report.approved_for_merge}",
        "",
        "## Static Scanner Output",
        f"{fence}text\n{state['static_scan_output']}\n{fence}",
        "",
        "## Findings",
        *(f"- {item}" for item in (report.findings or ["No blocking issues found."])),
        "",
        "## Remediation Summary",
        report.remediation_summary,
    ]
    extract_and_write_files(
        workspace,
        f"FILE: docs/SECURITY_REVIEW.md\n{fence}markdown\n" + "\n".join(md_lines) + f"\n{fence}",
    )
    return {"audit_report": report}


def build_review_graph(checkpointer=None):
    workflow = StateGraph(ReviewState)
    workflow.add_node("static_scanner", static_security_scanner_node)
    workflow.add_node("appsec_reviewer", appsec_and_quality_reviewer_agent)

    workflow.add_edge(START, "static_scanner")
    workflow.add_edge("static_scanner", "appsec_reviewer")
    workflow.add_edge("appsec_reviewer", END)
    return workflow.compile(checkpointer=checkpointer)


@mcp.tool()
async def review_and_audit_codebase(
    review_id: str = "",
    workspace_path: str = ".",
) -> str:
    """Executes Phase 3 of SDLC using dynamic agents.md & skills.md specs and concurrent CPU scanners."""
    root = Path(workspace_path).resolve()
    uid = review_id.strip() or uuid.uuid4().hex[:8]
    thread_id = f"phase3-{uid}"

    async with get_workspace_lock(root):
        arch_file = root / "docs" / "architecture.md"
        initial_state: ReviewState = {
            "workspace_path": str(root),
            "codebase_snapshot": get_source_snapshot(root),
            "arch_text": arch_file.read_text(encoding="utf-8") if arch_file.exists() else "",
            "static_scan_output": "",
            "audit_report": None,
        }

        config = {"configurable": {"thread_id": thread_id}}
        try:
            if AsyncSqliteSaver is not None:
                os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
                db_path = root / ".orchestrator" / "checkpoints.sqlite"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                    final_state = await build_review_graph(checkpointer=saver).ainvoke(initial_state, config=config)
            else:
                final_state = await asyncio.to_thread(
                    build_review_graph(checkpointer=get_sqlite_checkpointer(root)).invoke, initial_state, config
                )
        except TokenBudgetExceededError as e:
            log_audit_event(root, "phase3_review", thread_id, "TOKEN_BUDGET_HALTED", 1, {"error": str(e)})
            return f"Phase 3 HALTED by token circuit breaker: {e}"

        report = final_state.get("audit_report")
        status = "APPROVED" if (report and report.approved_for_merge) else "CHANGES_REQUESTED"

        log_audit_event(root, "phase3_review", thread_id, status, 1, {"risk_level": report.risk_level if report else "UNKNOWN"})
        return f"Phase 3 Review {status} (Risk: {report.risk_level if report else 'UNKNOWN'}). Saved to {root}/docs/SECURITY_REVIEW.md."


if __name__ == "__main__":
    if os.getenv("MCP_TRANSPORT", "stdio").lower() == "http":
        mcp.run(transport="streamable-http", host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8003")))
    else:
        mcp.run()