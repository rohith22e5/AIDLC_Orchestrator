import os
import sys
from pathlib import Path
from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from fastmcp import FastMCP

# Model imports (choose your provider)
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. INITIALIZE FASTMCP SERVER & MODEL
# =====================================================================

mcp = FastMCP("SDLC-Documentation-Orchestrator")

# Select your cloud model
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.1)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)

# =====================================================================
# 2. STRUCTURED SCHEMAS
# =====================================================================

class RequirementItem(BaseModel):
    id: str = Field(description="Unique ID e.g., FR-01, NFR-01")
    title: str = Field(description="Short functional title")
    description: str = Field(description="Detailed requirement description")
    acceptance_summary: str = Field(description="Core condition of satisfaction")

class RequirementsPayload(BaseModel):
    functional_requirements: List[RequirementItem]
    non_functional_requirements: List[RequirementItem]
    markdown_content: str = Field(description="Full requirements.md document")

class ArchitecturePayload(BaseModel):
    tech_stack: List[str] = Field(description="Primary libraries, databases, and frameworks")
    covered_fr_ids: List[str] = Field(description="List of FR IDs explicitly designed for")
    mermaid_topology: str = Field(description="Mermaid.js diagram for architecture/flow")
    mermaid_erd: str = Field(description="Mermaid.js diagram for database ERD")
    markdown_content: str = Field(
        description="Full architecture.md and spec.md document with embedded Mermaid code blocks"
    )

class UserStoryItem(BaseModel):
    id: str = Field(description="Story ID e.g., US-01")
    mapped_fr_id: str = Field(description="Traceability link to exact FR ID")
    story: str = Field(description="As a... I want... So that...")
    acceptance_criteria: List[str] = Field(description="Gherkin syntax (Given, When, Then)")

class PlanPayload(BaseModel):
    stories: List[UserStoryItem]
    milestones: List[str] = Field(description="Milestone phases e.g., Phase 1 MVP, Phase 2")
    mermaid_gantt: str = Field(description="Mermaid.js gantt or milestone flowchart")
    markdown_content: str = Field(
        description="Full plan.md and user_stories.md document with embedded Mermaid Gantt/Flow"
    )

class AuditFinding(BaseModel):
    severity: str = Field(description="'BLOCKER' or 'INFO'")
    target_agent: str = Field(description="'analyst', 'architect', or 'pm'")
    issue: str
    remediation: str

class QualityAuditPayload(BaseModel):
    is_approved: bool = Field(description="True if all traceability and quality checks pass")
    findings: List[AuditFinding] = Field(default_factory=list)
    actionable_critique: str = Field(
        description="Concise instructions for the agents to remediate discrepancies"
    )

class DocGenState(TypedDict):
    project_prompt: str
    target_dir: str
    requirements: Optional[RequirementsPayload]
    architecture: Optional[ArchitecturePayload]
    plan: Optional[PlanPayload]
    audit: Optional[QualityAuditPayload]
    iteration_count: int

# =====================================================================
# 3. AGENT NODES (Using stderr to avoid breaking MCP stdio)
# =====================================================================

def requirements_analyst(state: DocGenState):
    sys.stderr.write("\n[Agent 1] Analyst generating requirements...\n")
    critique_context = ""
    if state.get("audit") and not state["audit"].is_approved:
        critique_context = f"\nPREVIOUS AUDIT CRITIQUE TO RESOLVE:\n{state['audit'].actionable_critique}"

    prompt = f"""
    You are a Principal Business Analyst and Requirements Engineer.
    Project Brief:
    "{state['project_prompt']}"
    {critique_context}

    Deconstruct this into an exhaustive, production-grade specification.
    1. Extract Functional Requirements (FR-01, FR-02...) covering every user journey and edge case.
    2. Extract Non-Functional Requirements (NFR-01...) for latency, security, scale, availability, and auth.
    3. Generate the complete, professional Markdown document for `requirements.md`.
    """
    response = llm.with_structured_output(RequirementsPayload).invoke(prompt)
    return {"requirements": response}

def system_architect(state: DocGenState):
    sys.stderr.write("[Agent 2] Architect designing topology & data models...\n")
    reqs = state["requirements"]
    critique_context = ""
    if state.get("audit") and not state["audit"].is_approved:
        critique_context = f"\nPREVIOUS AUDIT CRITIQUE TO RESOLVE:\n{state['audit'].actionable_critique}"

    prompt = f"""
    You are a Principal Enterprise Systems Architect.
    Base Requirements:
    {reqs.markdown_content}
    {critique_context}

    Design the system architecture, component topology, and database models.
    Mandatory:
    1. Direct Traceability: Explicitly list all `covered_fr_ids` matching the requirements.
    2. Visual Artifact 1 (System Topology): Valid `graph TD` or `sequenceDiagram` in Mermaid.js.
    3. Visual Artifact 2 (Data Modeling): Valid `erDiagram` in Mermaid.js detailing tables/collections, keys, and relations.
    4. Provide API endpoint contracts, schemas, and auth flow.
    5. Generate the complete Markdown document for `architecture.md`. Embed both Mermaid diagrams inside ```mermaid code fences.
    """
    response = llm.with_structured_output(ArchitecturePayload).invoke(prompt)
    return {"architecture": response}

def project_manager(state: DocGenState):
    sys.stderr.write("[Agent 3] Agile PM breaking down user stories & sprints...\n")
    reqs = state["requirements"]
    arch = state["architecture"]
    critique_context = ""
    if state.get("audit") and not state["audit"].is_approved:
        critique_context = f"\nPREVIOUS AUDIT CRITIQUE TO RESOLVE:\n{state['audit'].actionable_critique}"

    prompt = f"""
    You are a Lead Agile Technical Program Manager.
    Requirements:
    {reqs.markdown_content}
    Architecture:
    {arch.markdown_content}
    {critique_context}

    Deconstruct into actionable sprints and user stories.
    Mandatory:
    1. Traceability: Every single User Story MUST explicitly reference its parent `mapped_fr_id`.
    2. Acceptance Criteria: Every User Story must use Gherkin syntax (Given... When... Then...).
    3. Visual Artifact: Valid Mermaid.js `gantt` chart or `graph LR` depicting delivery phases.
    4. Generate the complete Markdown document for `plan.md`. Embed the Mermaid visualization inside ```mermaid code fences.
    """
    response = llm.with_structured_output(PlanPayload).invoke(prompt)
    return {"plan": response}

def quality_auditor(state: DocGenState):
    sys.stderr.write("[Auditor] Evaluating quality gate & traceability...\n")
    reqs = state["requirements"]
    arch = state["architecture"]
    plan = state["plan"]
    
    fr_ids = {r.id for r in reqs.functional_requirements}
    arch_covered = set(arch.covered_fr_ids)
    story_covered = {s.mapped_fr_id for s in plan.stories}

    prompt = f"""
    You are a Senior Technical Auditor.
    Evaluate consistency and completeness:
    - Defined FR IDs: {sorted(list(fr_ids))}
    - Covered in Architecture: {sorted(list(arch_covered))}
    - Covered in User Stories: {sorted(list(story_covered))}
    
    Full Documents:
    [Requirements]
    {reqs.markdown_content}
    [Architecture]
    {arch.markdown_content}
    [Plan]
    {plan.markdown_content}

    Validation Rules:
    - Every FR ID must have corresponding architectural design and at least one user story.
    - Mermaid syntax must be structurally valid and complete.
    - If valid, approve (`is_approved = True`).
    - If major gaps exist, reject (`is_approved = False`), assign findings, and provide actionable critique.
    """
    response = llm.with_structured_output(QualityAuditPayload).invoke(prompt)
    new_iteration = state.get("iteration_count", 0) + 1
    sys.stderr.write(f"Audit decision: {'APPROVED' if response.is_approved else 'REJECTED'}\n")
    return {"audit": response, "iteration_count": new_iteration}

def file_writer(state: DocGenState):
    out_dir = Path(state.get("target_dir", "./docs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir / "requirements.md").write_text(state["requirements"].markdown_content, encoding="utf-8")
    (out_dir / "architecture.md").write_text(state["architecture"].markdown_content, encoding="utf-8")
    (out_dir / "plan.md").write_text(state["plan"].markdown_content, encoding="utf-8")
    
    sys.stderr.write(f"Files written to {out_dir.resolve()}\n")
    return state

# =====================================================================
# 4. GRAPH ASSEMBLY & CONDITIONAL ROUTING
# =====================================================================

def evaluate_audit_decision(state: DocGenState):
    audit = state.get("audit")
    iterations = state.get("iteration_count", 0)
    
    if audit and audit.is_approved:
        return "approved"
    if iterations >= 3:
        sys.stderr.write(f"Max audit cycles reached ({iterations}). Proceeding to write files.\n")
        return "approved"
        
    target_agents = {f.target_agent for f in audit.findings}
    if "analyst" in target_agents:
        return "retry_analyst"
    elif "architect" in target_agents:
        return "retry_architect"
    return "retry_pm"

workflow = StateGraph(DocGenState)

workflow.add_node("analyst", requirements_analyst)
workflow.add_node("architect", system_architect)
workflow.add_node("pm", project_manager)
workflow.add_node("auditor", quality_auditor)
workflow.add_node("writer", file_writer)

workflow.add_edge(START, "analyst")
workflow.add_edge("analyst", "architect")
workflow.add_edge("architect", "pm")
workflow.add_edge("pm", "auditor")

workflow.add_conditional_edges(
    "auditor",
    evaluate_audit_decision,
    {
        "approved": "writer",
        "retry_analyst": "analyst",
        "retry_architect": "architect",
        "retry_pm": "pm",
    }
)
workflow.add_edge("writer", END)

orchestrator = workflow.compile()

# =====================================================================
# 5. MCP TOOL REGISTRATION & SERVER ENTRY POINT
# =====================================================================

@mcp.tool()
def generate_project_specifications(project_brief: str, workspace_path: str = ".") -> str:
    """
    Runs the multi-agent orchestration pipeline to generate production-grade 
    requirements.md, architecture.md (with Mermaid diagrams), and plan.md (with Gherkin stories and Gantt).
    
    Args:
        project_brief: Comprehensive overview of the project, features, user flows, and tech constraints.
        workspace_path: Target root directory where the /docs folder will be saved.
    """
    initial_state: DocGenState = {
        "project_prompt": project_brief.strip(),
        "target_dir": workspace_path,
        "requirements": None,
        "architecture": None,
        "plan": None,
        "audit": None,
        "iteration_count": 0
    }
    
    orchestrator.invoke(initial_state)
    return f"Successfully generated requirements.md, architecture.md, and plan.md in {workspace_path}/docs"

if __name__ == "__main__":
    mcp.run()