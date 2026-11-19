import asyncio
import hashlib
import os
import sys
from pathlib import Path
from typing import List
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
    extract_and_write_files,
    get_llm,
    get_source_snapshot,
    get_sqlite_checkpointer,
    get_workspace_lock,
    log_audit_event,
)

# =====================================================================
# 1. INITIALIZE FASTMCP SERVER & MODEL
# =====================================================================

mcp = FastMCP("SDLC-Requirements-Architecture-Orchestrator")


# =====================================================================
# 2. SCHEMAS & STATE
# =====================================================================

class AmbiguityCheck(BaseModel):
    is_clear: bool = Field(description="True if requirements have enough detail to design architecture without guessing")
    clarifying_questions: List[str] = Field(description="Up to 3 concise, high-impact questions if critical specs are missing")


class SpecState(TypedDict):
    workspace_path: str
    product_prompt: str
    human_clarifications: str
    existing_codebase: str
    requirements_md: str
    architecture_md: str
    plan_md: str


# =====================================================================
# 3. AGENT NODES (WITH HITL INTERRUPT GATE)
# =====================================================================

def clarify_requirements_node(state: SpecState):
    """Checks for critical ambiguity before drafting specs; pauses via interrupt() if human input is needed."""
    if state.get("human_clarifications"):
        return {}

    sys.stderr.write("\n[Phase 1 -> Gate] Checking product prompt for critical ambiguities...\n")
    llm = get_llm()
    check_prompt = f"""
    You are a Principal Product Manager evaluating whether a feature request can be specified cleanly without guessing.

    [Product Request]
    {state['product_prompt']}

    [Existing Codebase Context]
    {state['existing_codebase'][:2000]}

    Only mark `is_clear=False` if critical architectural choices (e.g., auth mechanism, persistence engine, eviction policy, API format)
    are completely unspecified and cannot be inferred from the existing codebase. Ask at most 3 short, direct questions.
    """
    assessment = llm.with_structured_output(AmbiguityCheck).invoke(check_prompt)
    if not assessment.is_clear and assessment.clarifying_questions:
        sys.stderr.write(f"[Phase 1 -> HITL] Pausing for human clarification: {assessment.clarifying_questions}\n")
        human_reply = interrupt({
            "reason": "Clarification required before generating requirements",
            "questions": assessment.clarifying_questions,
        })
        return {"human_clarifications": str(human_reply)}

    return {}


def product_manager_agent(state: SpecState):
    sys.stderr.write("\n[Phase 1 -> Agent 1] Product Manager generating requirements & user stories...\n")
    llm = get_llm()
    fence = "`" * 3
    clarification_block = (
        f"\n[Human Clarifications]\n{state['human_clarifications']}\n"
        if state.get("human_clarifications")
        else ""
    )

    prompt = f"""
    You are a Principal Technical Product Manager.
    Convert the product request below into a crisp, testable Product Requirements Document (`docs/requirements.md`).

    [Product Request]
    {state['product_prompt']}
    {clarification_block}
    [Existing Codebase Context (if any)]
    {state['existing_codebase'][:3000]}

    Include:
    1. Functional & Non-Functional Requirements
    2. Prioritized User Stories (`US-01`, `US-02`, ...) with explicit Given/When/Then acceptance criteria
    3. Edge Cases & Input Validation Contracts (exact exception types & error messages)

    Output strictly using this Markdown format:

    FILE: docs/requirements.md
    {fence}markdown
    # Product Requirements Document
    ...
    {fence}
    """
    res = llm.invoke(prompt)
    workspace = Path(state["workspace_path"])
    extract_and_write_files(workspace, res.content)
    req_file = workspace / "docs" / "requirements.md"
    return {"requirements_md": req_file.read_text(encoding="utf-8") if req_file.exists() else res.content}


def system_architect_agent(state: SpecState):
    sys.stderr.write("\n[Phase 1 -> Agent 2] System Architect designing technical architecture...\n")
    llm = get_llm()
    fence = "`" * 3
    prompt = f"""
    You are a Principal Software Architect.
    Design a clean, modular technical architecture (`docs/architecture.md`) satisfying the requirements below.

    [Requirements]
    {state['requirements_md'][:4000]}

    [Existing Codebase Context]
    {state['existing_codebase'][:2500]}

    Include:
    1. Module & Directory Layout (`src/`, `tests/`)
    2. Public Class/Function Signatures & Type Contracts
    3. Concurrency, Thread-Safety (`threading.RLock`), and Error Handling Strategy
    4. Mermaid Component/Sequence Diagram

    Output strictly using this Markdown format:

    FILE: docs/architecture.md
    {fence}markdown
    # System Architecture
    ...
    {fence}
    """
    res = llm.invoke(prompt)
    workspace = Path(state["workspace_path"])
    extract_and_write_files(workspace, res.content)
    arch_file = workspace / "docs" / "architecture.md"
    return {"architecture_md": arch_file.read_text(encoding="utf-8") if arch_file.exists() else res.content}


def tech_lead_planner_agent(state: SpecState):
    sys.stderr.write("\n[Phase 1 -> Agent 3] Tech Lead creating phased implementation plan...\n")
    llm = get_llm()
    fence = "`" * 3
    prompt = f"""
    You are a Staff Engineering Tech Lead.
    Create a step-by-step TDD implementation plan (`docs/plan.md`) mapped directly to the user stories.

    [Requirements]
    {state['requirements_md'][:3000]}

    [Architecture]
    {state['architecture_md'][:3000]}

    Output strictly using this Markdown format:

    FILE: docs/plan.md
    {fence}markdown
    # Implementation Plan
    ...
    {fence}
    """
    res = llm.invoke(prompt)
    workspace = Path(state["workspace_path"])
    extract_and_write_files(workspace, res.content)
    plan_file = workspace / "docs" / "plan.md"
    return {"plan_md": plan_file.read_text(encoding="utf-8") if plan_file.exists() else res.content}


# =====================================================================
# 4. GRAPH COMPILATION HELPER
# =====================================================================

def build_spec_graph(checkpointer=None):
    workflow = StateGraph(SpecState)
    workflow.add_node("clarify_gate", clarify_requirements_node)
    workflow.add_node("pm_agent", product_manager_agent)
    workflow.add_node("architect_agent", system_architect_agent)
    workflow.add_node("planner_agent", tech_lead_planner_agent)

    workflow.add_edge(START, "clarify_gate")
    workflow.add_edge("clarify_gate", "pm_agent")
    workflow.add_edge("pm_agent", "architect_agent")
    workflow.add_edge("architect_agent", "planner_agent")
    workflow.add_edge("planner_agent", END)
    return workflow.compile(checkpointer=checkpointer)


# =====================================================================
# 5. MCP TOOLS (START & RESUME)
# =====================================================================

@mcp.tool()
async def generate_project_specs(
    product_prompt: str,
    skip_clarification: bool = False,
    force_regenerate: bool = False,
    workspace_path: str = ".",
) -> str:
    """
    Executes Phase 1 of SDLC: Generates docs/requirements.md, docs/architecture.md, and docs/plan.md.
    If critical details are ambiguous (and skip_clarification=False), pauses and returns clarifying
    questions with a thread_id so the user can answer via `resume_project_specs`.
    """
    root = Path(workspace_path).resolve()
    prompt_hash = hashlib.sha256(product_prompt.strip().encode("utf-8")).hexdigest()[:12]
    thread_id = f"phase1-{prompt_hash}"

    async with get_workspace_lock(root):
        cache_marker = root / ".orchestrator" / f"spec_{prompt_hash}.done"
        req_exists = (root / "docs" / "requirements.md").exists()
        arch_exists = (root / "docs" / "architecture.md").exists()
        plan_exists = (root / "docs" / "plan.md").exists()

        if not force_regenerate and cache_marker.exists() and req_exists and arch_exists and plan_exists:
            log_audit_event(root, "phase1_specs", thread_id, "CACHE_HIT_SKIPPED", 0, {"hash": prompt_hash})
            return f"Phase 1 Specs CACHE HIT (0 tokens spent) for hash [{prompt_hash}]."

        initial_state: SpecState = {
            "workspace_path": str(root),
            "product_prompt": product_prompt.strip(),
            "human_clarifications": "Proceed with sensible engineering defaults." if skip_clarification else "",
            "existing_codebase": get_source_snapshot(root, max_chars=4000),
            "requirements_md": "",
            "architecture_md": "",
            "plan_md": "",
        }

        config = {"configurable": {"thread_id": thread_id}}
        try:
            if AsyncSqliteSaver is not None:
                os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
                db_path = root / ".orchestrator" / "checkpoints.sqlite"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                    graph = build_spec_graph(checkpointer=saver)
                    result = await graph.ainvoke(initial_state, config=config)
            else:
                graph = build_spec_graph(checkpointer=get_sqlite_checkpointer(root))
                result = await asyncio.to_thread(graph.invoke, initial_state, config)
        except TokenBudgetExceededError as e:
            log_audit_event(root, "phase1_specs", thread_id, "TOKEN_BUDGET_HALTED", 1, {"error": str(e)})
            return f"Phase 1 HALTED by token circuit breaker: {e}"

        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            questions = "\n".join(f"- {q}" for q in payload.get("questions", []))
            log_audit_event(root, "phase1_specs", thread_id, "AWAITING_HUMAN_INPUT", 0, {"questions": payload.get("questions", [])})
            return (
                f"Phase 1 PAUSED for clarification (thread_id='{thread_id}'):\n{questions}\n\n"
                f"Reply by calling `resume_project_specs(thread_id='{thread_id}', clarifications='...')`."
            )

        cache_marker.parent.mkdir(parents=True, exist_ok=True)
        cache_marker.write_text(prompt_hash, encoding="utf-8")
        log_audit_event(root, "phase1_specs", thread_id, "SPECS_GENERATED", 1, {"hash": prompt_hash})
        return f"Phase 1 Specs GENERATED [{prompt_hash}]: Saved docs/requirements.md, docs/architecture.md, and docs/plan.md."


@mcp.tool()
async def resume_project_specs(
    thread_id: str,
    clarifications: str,
    workspace_path: str = ".",
) -> str:
    """Resumes a paused Phase 1 specification workflow with the human's answers."""
    root = Path(workspace_path).resolve()
    config = {"configurable": {"thread_id": thread_id}}

    async with get_workspace_lock(root):
        try:
            if AsyncSqliteSaver is not None:
                os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
                db_path = root / ".orchestrator" / "checkpoints.sqlite"
                async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                    graph = build_spec_graph(checkpointer=saver)
                    await graph.ainvoke(Command(resume=clarifications.strip()), config=config)
            else:
                graph = build_spec_graph(checkpointer=get_sqlite_checkpointer(root))
                await asyncio.to_thread(graph.invoke, Command(resume=clarifications.strip()), config)
        except TokenBudgetExceededError as e:
            log_audit_event(root, "phase1_specs", thread_id, "TOKEN_BUDGET_HALTED", 1, {"error": str(e)})
            return f"Phase 1 HALTED by token circuit breaker: {e}"

        log_audit_event(root, "phase1_specs", thread_id, "SPECS_RESUMED_AND_GENERATED", 1)
        return f"Phase 1 Specs RESUMED & GENERATED for [{thread_id}]: Saved docs/requirements.md, docs/architecture.md, and docs/plan.md."


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "http":
        mcp.run(
            transport="streamable-http",
            host=os.getenv("MCP_HOST", "127.0.0.1"),
            port=int(os.getenv("MCP_PORT", "8001")),
        )
    else:
        mcp.run()