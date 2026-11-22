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
    load_agent_prompt,
    log_audit_event,
)

mcp = FastMCP("SDLC-Requirements-Architecture-Orchestrator")


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


def _write_doc_fallback(workspace: Path, rel_path: str, raw_response: str) -> str:
    """
    Tries `extract_and_write_files()` first; if the model omitted the `FILE:` fenced wrapper
    (common with nested Markdown in PRDs), writes `raw_response` directly to `rel_path`.
    """
    written = extract_and_write_files(workspace, raw_response, allowed_prefixes=("docs/",))
    target = workspace / rel_path
    if written:
        return written[0].content
    clean_text = raw_response.strip() + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(clean_text, encoding="utf-8")
    sys.stderr.write(f"[Phase 1 -> Fallback] Wrote {rel_path} directly ({len(clean_text)} chars)\n")
    return clean_text


def clarify_requirements_node(state: SpecState):
    if state.get("human_clarifications"):
        return {}

    sys.stderr.write("\n[Phase 1 -> Gate] Checking product prompt for critical ambiguities...\n")
    workspace = Path(state["workspace_path"])
    llm = get_llm("clarify_gate")
    system_spec = load_agent_prompt("clarify_gate", workspace=workspace)

    check_prompt = f"""{system_spec}

[Product Request]
{state['product_prompt']}

[Existing Codebase Context]
{state['existing_codebase'][:2000]}"""

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
    workspace = Path(state["workspace_path"])
    llm = get_llm("pm_agent")
    system_spec = load_agent_prompt("pm_agent", ["markdown_file_output"], workspace=workspace)
    clarification_block = f"\n[Human Clarifications]\n{state['human_clarifications']}\n" if state.get("human_clarifications") else ""

    prompt = f"""{system_spec}

[Product Request]
{state['product_prompt']}
{clarification_block}
[Existing Codebase Context]
{state['existing_codebase'][:3000]}

Target Output File: `docs/requirements.md`"""

    res = llm.invoke(prompt)
    req_md = _write_doc_fallback(workspace, "docs/requirements.md", res.content)
    return {"requirements_md": req_md}


def system_architect_agent(state: SpecState):
    sys.stderr.write("\n[Phase 1 -> Agent 2] System Architect designing technical architecture...\n")
    workspace = Path(state["workspace_path"])
    llm = get_llm("architect_agent")
    system_spec = load_agent_prompt("architect_agent", ["markdown_file_output"], workspace=workspace)

    prompt = f"""{system_spec}

[Requirements]
{state['requirements_md'][:4000]}

[Existing Codebase Context]
{state['existing_codebase'][:2500]}

Target Output File: `docs/architecture.md`"""

    res = llm.invoke(prompt)
    arch_md = _write_doc_fallback(workspace, "docs/architecture.md", res.content)
    return {"architecture_md": arch_md}


def tech_lead_planner_agent(state: SpecState):
    sys.stderr.write("\n[Phase 1 -> Agent 3] Tech Lead creating phased implementation plan...\n")
    workspace = Path(state["workspace_path"])
    llm = get_llm("planner_agent")
    system_spec = load_agent_prompt("planner_agent", ["markdown_file_output"], workspace=workspace)

    prompt = f"""{system_spec}

[Requirements]
{state['requirements_md'][:3000]}

[Architecture]
{state['architecture_md'][:3000]}

Target Output File: `docs/plan.md`"""

    res = llm.invoke(prompt)
    plan_md = _write_doc_fallback(workspace, "docs/plan.md", res.content)
    return {"plan_md": plan_md}


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


@mcp.tool()
async def generate_project_specs(
    product_prompt: str,
    skip_clarification: bool = False,
    force_regenerate: bool = False,
    workspace_path: str = ".",
) -> str:
    """Executes Phase 1 of SDLC using dynamic agents.md & skills.md specs with HITL clarification."""
    root = Path(workspace_path).resolve()
    prompt_hash = hashlib.sha256(product_prompt.strip().encode("utf-8")).hexdigest()[:12]
    thread_id = f"phase1-{prompt_hash}"

    async with get_workspace_lock(root):
        cache_marker = root / ".orchestrator" / f"spec_{prompt_hash}.done"
        if (
            not force_regenerate
            and cache_marker.exists()
            and (root / "docs" / "requirements.md").exists()
            and (root / "docs" / "architecture.md").exists()
            and (root / "docs" / "plan.md").exists()
        ):
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
                    result = await build_spec_graph(checkpointer=saver).ainvoke(initial_state, config=config)
            else:
                result = await asyncio.to_thread(
                    build_spec_graph(checkpointer=get_sqlite_checkpointer(root)).invoke, initial_state, config
                )
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
                    await build_spec_graph(checkpointer=saver).ainvoke(Command(resume=clarifications.strip()), config=config)
            else:
                await asyncio.to_thread(
                    build_spec_graph(checkpointer=get_sqlite_checkpointer(root)).invoke,
                    Command(resume=clarifications.strip()),
                    config,
                )
        except TokenBudgetExceededError as e:
            log_audit_event(root, "phase1_specs", thread_id, "TOKEN_BUDGET_HALTED", 1, {"error": str(e)})
            return f"Phase 1 HALTED by token circuit breaker: {e}"

        log_audit_event(root, "phase1_specs", thread_id, "SPECS_RESUMED_AND_GENERATED", 1)
        return f"Phase 1 Specs RESUMED & GENERATED for [{thread_id}]: Saved docs/requirements.md, docs/architecture.md, and docs/plan.md."


if __name__ == "__main__":
    if os.getenv("MCP_TRANSPORT", "stdio").lower() == "http":
        mcp.run(transport="streamable-http", host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8001")))
    else:
        mcp.run()