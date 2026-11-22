import asyncio
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from fastmcp import FastMCP

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:
    AsyncSqliteSaver = None

from orchestrator_core import (
    TokenBudgetExceededError,
    get_llm,
    get_source_snapshot,
    get_sqlite_checkpointer,
    get_workspace_lock,
    git_worktree_sandbox,
    log_audit_event,
)

mcp = FastMCP("SDLC-Master-Autonomous-Orchestrator")


# =====================================================================
# 1. DYNAMICALLY LOAD PHASE SUB-GRAPHS FROM EXISTING ORCHESTRATORS
# =====================================================================

def _load_module(filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(filename.replace("-", "_").replace(".py", ""), str(path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    return mod


_req_mod = _load_module("requirement-orchestrator.py")
_cod_mod = _load_module("coding-orchestrator.py")
_rev_mod = _load_module("review-orchestrator.py")
_int_mod = _load_module("integration-orchestrator.py")
_dev_mod = _load_module("devops-orchestrator.py")
_sre_mod = _load_module("self-healing-orchestrator.py")


# =====================================================================
# 2. MASTER STATE & STORY EXTRACTOR SCHEMA
# =====================================================================

class ExtractedStory(BaseModel):
    story_id: str = Field(description="e.g. US-01")
    story_spec: str = Field(description="Full user story title + Given/When/Then acceptance criteria")


class StoryBacklog(BaseModel):
    stories: List[ExtractedStory] = Field(description="Ordered list of user stories parsed from docs/requirements.md")


class MasterSDLCState(TypedDict):
    workspace_path: str
    product_prompt: str
    human_clarifications: str
    architecture_approved: bool
    release_approved: bool

    requirements_md: str
    architecture_md: str
    plan_md: str
    stories: List[Dict[str, str]]
    completed_stories: List[str]
    coding_failed: bool

    security_status: str
    security_blocked: bool
    e2e_status: str
    devops_status: str
    summary_log: List[str]


# =====================================================================
# 3. MASTER GRAPH NODES (ASYNC SUBGRAPHS + HUMAN DECISION GATES)
# =====================================================================

async def phase1_specs_and_clarify_node(state: MasterSDLCState):
    """Runs Phase 1 (Ambiguity Check -> PM -> Architect -> Planner) and extracts user stories."""
    root = Path(state["workspace_path"])
    spec_graph = _req_mod.build_spec_graph()

    sub_state = {
        "workspace_path": str(root),
        "product_prompt": state["product_prompt"],
        "human_clarifications": state.get("human_clarifications", ""),
        "existing_codebase": get_source_snapshot(root, max_chars=4000),
        "requirements_md": "",
        "architecture_md": "",
        "plan_md": "",
    }
    out = await spec_graph.ainvoke(sub_state)

    llm = get_llm("planner_agent")
    backlog = llm.with_structured_output(StoryBacklog).invoke(
        f"Extract all user stories (`US-01`, `US-02`, ...) with their acceptance criteria from:\n\n{out['requirements_md'][:4500]}"
    )
    stories_list = [{"story_id": s.story_id, "story_spec": s.story_spec} for s in backlog.stories]
    if not stories_list:
        stories_list = [{"story_id": "US-01", "story_spec": out["requirements_md"][:2000]}]

    logs = list(state.get("summary_log", []))
    logs.append(f"Phase 1 Complete: Generated specs & extracted {len(stories_list)} user stor(ies).")
    return {
        "requirements_md": out["requirements_md"],
        "architecture_md": out["architecture_md"],
        "plan_md": out["plan_md"],
        "stories": stories_list,
        "summary_log": logs,
    }


def human_architecture_gate_node(state: MasterSDLCState):
    """Pauses for human confirmation of Architecture & User Story plan before writing code."""
    if state.get("architecture_approved"):
        return {}

    story_titles = [f"{s['story_id']}: {s['story_spec'].splitlines()[0][:80]}" for s in state.get("stories", [])]
    decision = interrupt({
        "gate": "ARCHITECTURE_AND_BACKLOG_APPROVAL",
        "message": "Review `docs/requirements.md`, `docs/architecture.md`, and `docs/plan.md`. Approve to begin autonomous coding, review, E2E, and DevOps?",
        "planned_stories": story_titles,
    })
    reply = str(decision).strip()
    logs = list(state.get("summary_log", []))
    logs.append(f"Human Gate (Architecture): {reply}")
    return {
        "architecture_approved": True,
        "human_clarifications": f"{state.get('human_clarifications', '')}\n[Architecture Feedback]: {reply}".strip(),
        "summary_log": logs,
    }


async def phase2_tdd_stories_node(state: MasterSDLCState):
    """Executes Phase 2 TDD coding inside isolated Git worktrees for each story, merging verified branches."""
    root = Path(state["workspace_path"])
    coding_graph = _cod_mod.build_coding_graph()
    completed = list(state.get("completed_stories", []))
    logs = list(state.get("summary_log", []))
    any_failed = False

    for story in state.get("stories", []):
        sid = story["story_id"]
        if sid in completed:
            continue

        with git_worktree_sandbox(root, f"phase2-{sid}") as sandbox_dir:
            sub_state = {
                "workspace_path": str(sandbox_dir),
                "story_id": sid,
                "story_spec": story["story_spec"],
                "arch_text": state.get("architecture_md", ""),
                "plan_text": state.get("plan_md", ""),
                "test_runner_command": "",
                "codebase_snapshot": get_source_snapshot(sandbox_dir),
                "written_files": [],
                "last_test_stdout": "",
                "last_test_stderr": "",
                "last_test_exit_code": 1,
                "iteration_count": 0,
            }
            out = await coding_graph.ainvoke(sub_state)
            passed = out.get("last_test_exit_code", 1) == 0

            if passed and (root / ".git").exists() and sandbox_dir != root:
                safe_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", f"phase2-{sid}").strip("-")
                branch_name = f"sdlc/{safe_slug}"
                subprocess.run(["git", "add", "-A"], cwd=sandbox_dir, capture_output=True)
                subprocess.run(["git", "commit", "-m", f"feat({sid}): verified TDD implementation"], cwd=sandbox_dir, capture_output=True)
                subprocess.run(["git", "merge", "--no-ff", branch_name, "-m", f"merge({sid}): verified story"], cwd=root, capture_output=True)

            if passed:
                status_txt = "VERIFIED & MERGED"
                completed.append(sid)
            else:
                any_failed = True
                status_txt = f"FAILED_MAX_RETRIES ({out.get('last_test_stderr', '')[:160]})"

            logs.append(f"Phase 2 [{sid}]: {status_txt} in {out.get('iteration_count', 0)} pass(es).")

    return {"completed_stories": completed, "coding_failed": any_failed, "summary_log": logs}


def coding_quality_gate_node(state: MasterSDLCState):
    """Pauses for human intervention if TDD unit tests failed max retries."""
    if not state.get("coding_failed"):
        return {}

    decision = interrupt({
        "gate": "TDD_VERIFICATION_FAILED_GATE",
        "message": "One or more stories failed pytest after 3 passes. Inspect the `.worktrees/` directory or reply 'override' to continue anyway.",
        "summary": state.get("summary_log", []),
    })
    logs = list(state.get("summary_log", []))
    logs.append(f"Human Gate (Coding Recovery): {str(decision).strip()}")
    return {"coding_failed": False, "summary_log": logs}


async def phase3_security_review_node(state: MasterSDLCState):
    """Runs Phase 3 concurrent static scans + AppSec audit."""
    root = Path(state["workspace_path"])
    review_graph = _rev_mod.build_review_graph()
    out = await review_graph.ainvoke({
        "workspace_path": str(root),
        "codebase_snapshot": get_source_snapshot(root),
        "arch_text": state.get("architecture_md", ""),
        "static_scan_output": "",
        "audit_report": None,
    })
    report = out.get("audit_report")
    blocked = bool(report and not report.approved_for_merge)
    status = f"{report.risk_level} (Approved={report.approved_for_merge})" if report else "COMPLETED"
    logs = list(state.get("summary_log", []))
    logs.append(f"Phase 3 AppSec & Review: {status}")
    return {"security_status": status, "security_blocked": blocked, "summary_log": logs}


def security_quality_gate_node(state: MasterSDLCState):
    """Pauses if AppSec review blocked merge due to HIGH/CRITICAL findings."""
    if not state.get("security_blocked"):
        return {}

    decision = interrupt({
        "gate": "SECURITY_REVIEW_BLOCKED_GATE",
        "message": "AppSec Reviewer flagged blocking issues in `docs/SECURITY_REVIEW.md`. Fix or reply 'override' to proceed to E2E.",
        "security_status": state.get("security_status", ""),
    })
    logs = list(state.get("summary_log", []))
    logs.append(f"Human Gate (Security Override): {str(decision).strip()}")
    return {"security_blocked": False, "summary_log": logs}


async def phase4_integration_node(state: MasterSDLCState):
    """Runs Phase 4 E2E integration tests."""
    root = Path(state["workspace_path"])
    int_graph = _int_mod.build_integration_graph()
    out = await int_graph.ainvoke({
        "workspace_path": str(root),
        "compose_project_name": f"sdlc_e2e_{hashlib.sha1(str(root).encode()).hexdigest()[:6]}",
        "arch_text": state.get("architecture_md", ""),
        "codebase_snapshot": get_source_snapshot(root),
        "written_files": [],
        "last_stdout": "",
        "last_stderr": "",
        "last_exit_code": 1,
        "iteration_count": 0,
    })
    status = "VERIFIED" if out.get("last_exit_code", 1) == 0 else "WARNINGS"
    logs = list(state.get("summary_log", []))
    logs.append(f"Phase 4 Integration & E2E: {status}")
    return {"e2e_status": status, "summary_log": logs}


async def phase5_devops_node(state: MasterSDLCState):
    """Runs Phase 5 Dockerfile, Compose, and CI/CD pipeline generation."""
    root = Path(state["workspace_path"])
    dev_graph = _dev_mod.build_devops_graph()
    out = await dev_graph.ainvoke({
        "workspace_path": str(root),
        "arch_text": state.get("architecture_md", ""),
        "codebase_snapshot": get_source_snapshot(root),
        "written_files": [],
        "validation_stderr": "",
        "validation_exit_code": 1,
        "iteration_count": 0,
    })
    status = "VALIDATED" if out.get("validation_exit_code", 1) == 0 else "GENERATED_WITH_WARNINGS"
    logs = list(state.get("summary_log", []))
    logs.append(f"Phase 5 DevOps & CI/CD: {status}")
    return {"devops_status": status, "summary_log": logs}


def human_release_gate_node(state: MasterSDLCState):
    """Final human confirmation gate before marking the application release-ready."""
    if state.get("release_approved"):
        return {}

    decision = interrupt({
        "gate": "FINAL_RELEASE_APPROVAL",
        "message": "Implementation, Security Review, E2E Integration, and DevOps artifacts are ready. Confirm final release sign-off?",
        "summary": state.get("summary_log", []),
    })
    logs = list(state.get("summary_log", []))
    logs.append(f"Human Gate (Release): {str(decision).strip()}")
    return {"release_approved": True, "summary_log": logs}


# =====================================================================
# 4. COMPILE MASTER GRAPH & REGISTER MCP TOOLS
# =====================================================================

def build_master_graph(checkpointer=None):
    wf = StateGraph(MasterSDLCState)
    wf.add_node("phase1_specs", phase1_specs_and_clarify_node)
    wf.add_node("human_arch_gate", human_architecture_gate_node)
    wf.add_node("phase2_coding", phase2_tdd_stories_node)
    wf.add_node("coding_gate", coding_quality_gate_node)
    wf.add_node("phase3_review", phase3_security_review_node)
    wf.add_node("security_gate", security_quality_gate_node)
    wf.add_node("phase4_integration", phase4_integration_node)
    wf.add_node("phase5_devops", phase5_devops_node)
    wf.add_node("human_release_gate", human_release_gate_node)

    wf.add_edge(START, "phase1_specs")
    wf.add_edge("phase1_specs", "human_arch_gate")
    wf.add_edge("human_arch_gate", "phase2_coding")
    wf.add_edge("phase2_coding", "coding_gate")
    wf.add_edge("coding_gate", "phase3_review")
    wf.add_edge("phase3_review", "security_gate")
    wf.add_edge("security_gate", "phase4_integration")
    wf.add_edge("phase4_integration", "phase5_devops")
    wf.add_edge("phase5_devops", "human_release_gate")
    wf.add_edge("human_release_gate", END)
    return wf.compile(checkpointer=checkpointer)


def _format_result(thread_id: str, result: Dict[str, Any]) -> str:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return (
            f"MASTER ORCHESTRATOR PAUSED at [{payload.get('gate', 'HUMAN_DECISION')}] (thread_id='{thread_id}'):\n"
            f"{payload}\n\n"
            f"Call `continue_autonomous_sdlc(thread_id='{thread_id}', human_response='...')` to proceed."
        )
    summary = "\n".join(f"- {line}" for line in result.get("summary_log", []))
    return f"MASTER SDLC PIPELINE COMPLETE (thread_id='{thread_id}'):\n{summary}"


@mcp.tool()
async def start_autonomous_sdlc(
    product_prompt: str,
    auto_approve_gates: bool = False,
    restart_fresh: bool = False,
    workspace_path: str = ".",
) -> str:
    """
    Runs the full end-to-end SDLC pipeline (Requirements -> Architecture -> TDD Coding ->
    AppSec Review -> E2E Integration -> DevOps CI/CD) from a single prompt, pausing at key
    human decision checkpoints unless `auto_approve_gates=True`.
    """
    root = Path(workspace_path).resolve()
    slug = hashlib.sha256(product_prompt.strip().encode("utf-8")).hexdigest()[:10]
    thread_id = f"master-{slug}"

    async with get_workspace_lock(root):
        initial_state: MasterSDLCState = {
            "workspace_path": str(root),
            "product_prompt": product_prompt.strip(),
            "human_clarifications": "Proceed with sensible engineering defaults." if auto_approve_gates else "",
            "architecture_approved": auto_approve_gates,
            "release_approved": auto_approve_gates,
            "requirements_md": "",
            "architecture_md": "",
            "plan_md": "",
            "stories": [],
            "completed_stories": [],
            "coding_failed": False,
            "security_status": "",
            "security_blocked": False,
            "e2e_status": "",
            "devops_status": "",
            "summary_log": [],
        }
        config = {"configurable": {"thread_id": thread_id}}
        try:
            if AsyncSqliteSaver is not None:
                os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
                db_path = root / ".orchestrator" / "checkpoints.sqlite"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                    existing_ckpt = None if restart_fresh else await saver.aget_tuple(config)
                    payload = None if (existing_ckpt and existing_ckpt.checkpoint) else initial_state
                    res = await build_master_graph(checkpointer=saver).ainvoke(payload, config=config)
            else:
                saver = get_sqlite_checkpointer(root)
                existing_ckpt = None if (restart_fresh or saver is None) else saver.get_tuple(config)
                payload = None if (existing_ckpt and existing_ckpt.checkpoint) else initial_state
                res = await asyncio.to_thread(build_master_graph(checkpointer=saver).invoke, payload, config)
        except TokenBudgetExceededError as e:
            log_audit_event(root, "master_sdlc", thread_id, "TOKEN_BUDGET_HALTED", 0, {"error": str(e)})
            return f"Master SDLC HALTED by token budget circuit breaker: {e}"

        status = "PAUSED_FOR_HUMAN" if "__interrupt__" in res else "COMPLETED"
        log_audit_event(root, "master_sdlc", thread_id, status, 1, {"summary": res.get("summary_log", [])})
        return _format_result(thread_id, res)


@mcp.tool()
async def continue_autonomous_sdlc(
    thread_id: str,
    human_response: str,
    workspace_path: str = ".",
) -> str:
    """
    Answers a clarification question or approves an Architecture/Release checkpoint to resume
    the Master SDLC pipeline from its exact SQLite checkpoint.
    """
    root = Path(workspace_path).resolve()
    config = {"configurable": {"thread_id": thread_id}}

    async with get_workspace_lock(root):
        try:
            if AsyncSqliteSaver is not None:
                os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
                db_path = root / ".orchestrator" / "checkpoints.sqlite"
                async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                    res = await build_master_graph(checkpointer=saver).ainvoke(
                        Command(resume=human_response.strip()), config=config
                    )
            else:
                res = await asyncio.to_thread(
                    build_master_graph(checkpointer=get_sqlite_checkpointer(root)).invoke,
                    Command(resume=human_response.strip()),
                    config,
                )
        except TokenBudgetExceededError as e:
            log_audit_event(root, "master_sdlc", thread_id, "TOKEN_BUDGET_HALTED", 0, {"error": str(e)})
            return f"Master SDLC HALTED by token budget circuit breaker: {e}"

        status = "PAUSED_FOR_HUMAN" if "__interrupt__" in res else "COMPLETED"
        log_audit_event(root, "master_sdlc", thread_id, status, 1, {"summary": res.get("summary_log", [])})
        return _format_result(thread_id, res)


@mcp.tool()
async def report_production_incident(
    incident_logs: str,
    incident_id: str = "inc-auto",
    workspace_path: str = ".",
) -> str:
    """Triggers Phase 6 SRE Triage, Regression Test, Isolated Worktree Hotfix & Post-Mortem."""
    return await _sre_mod.heal_runtime_incident(
        incident_logs=incident_logs,
        incident_id=incident_id,
        workspace_path=workspace_path,
    )


@mcp.tool()
async def run_agile_sprint(
    sprint_goal: str,
    sprint_number: int = 1,
    workspace_path: str = ".",
) -> str:
    """
    Executes a single Agile Sprint increment on top of the existing codebase.
    """
    prompt = (
        f"[Sprint {sprint_number} Increment]\n"
        f"Build ONLY the features required for this sprint goal on top of the existing repository:\n"
        f"{sprint_goal.strip()}"
    )
    return await start_autonomous_sdlc(
        product_prompt=prompt,
        auto_approve_gates=False,
        workspace_path=workspace_path,
    )


if __name__ == "__main__":
    if os.getenv("MCP_TRANSPORT", "stdio").lower() == "http":
        mcp.run(transport="streamable-http", host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8000")))
    else:
        mcp.run()