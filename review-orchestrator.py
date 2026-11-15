import os
import sys
from pathlib import Path
from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from fastmcp import FastMCP

# Switch between your preferred cloud model:
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END

# =====================================================================
# 1. INITIALIZE FASTMCP SERVER & MODEL
# =====================================================================

mcp = FastMCP("SDLC-Review-AppSec-Orchestrator")

# High-reasoning model for static analysis & security auditing
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0)
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)


# =====================================================================
# 2. STRUCTURED SCHEMAS
# =====================================================================

class SecurityQualityFinding(BaseModel):
    severity: str = Field(description="'CRITICAL', 'HIGH', 'MEDIUM', or 'LOW'")
    category: str = Field(description="'SECURITY', 'PERFORMANCE', 'ARCHITECTURE_DRIFT', or 'CODE_QUALITY'")
    file_path: str = Field(description="Relative path to the affected file")
    line_hint: str = Field(description="Function name or snippet where the issue occurs")
    description: str = Field(description="Detailed explanation of the vulnerability or anti-pattern")
    remediation_plan: str = Field(description="Exact code-level instructions to fix the issue")

class ReviewAuditPayload(BaseModel):
    is_approved: bool = Field(
        description="True ONLY if there are zero CRITICAL or HIGH severity findings."
    )
    findings: List[SecurityQualityFinding] = Field(default_factory=list)
    audit_markdown: str = Field(
        description="Complete, formatted SECURITY_AUDIT.md report summarizing findings, scores, and sign-off status"
    )

class FilePatch(BaseModel):
    file_path: str = Field(description="Relative path to file being patched")
    updated_content: str = Field(description="Complete updated file content with all fixes applied")

class RemediationOutput(BaseModel):
    patched_files: List[FilePatch]
    changelog_summary: str

class ReviewState(TypedDict):
    workspace_path: str
    target_dir: str
    arch_text: str
    requirements_text: str
    codebase_snapshot: str
    
    audit_report: Optional[ReviewAuditPayload]
    iteration_count: int


# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================

def collect_codebase(workspace: Path, target_subdir: str) -> str:
    """Reads all source files in the target directory into a structured context string."""
    search_root = workspace / target_subdir
    if not search_root.exists():
        search_root = workspace

    ignore_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", "docs", ".pytest_cache"}
    allowed_exts = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".sql"}
    
    snippets = []
    for path in sorted(search_root.rglob("*")):
        if any(part in ignore_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix in allowed_exts:
            rel_path = path.relative_to(workspace).as_posix()
            content = path.read_text(encoding="utf-8", errors="ignore")
            snippets.append(f"### FILE: {rel_path}\n```\n{content}\n```")
            
    return "\n\n".join(snippets)


# =====================================================================
# 4. AGENT NODES
# =====================================================================

def appsec_and_quality_auditor(state: ReviewState):
    sys.stderr.write("\n[Phase 3 -> Auditor] Scanning codebase for OWASP, secrets, performance & quality...\n")
    
    # Refresh codebase snapshot in case the remediation agent patched files
    workspace = Path(state["workspace_path"])
    current_code = collect_codebase(workspace, state["target_dir"])
    
    if not current_code.strip():
        sys.stderr.write("Warning: No source files found to audit.\n")
        empty_report = ReviewAuditPayload(
            is_approved=True,
            findings=[],
            audit_markdown="# Security & Quality Audit\n\nNo source files detected in target directory."
        )
        return {"audit_report": empty_report, "iteration_count": state.get("iteration_count", 0) + 1}

    prompt = f"""
    You are a Staff Application Security (AppSec) Engineer and Principal Code Reviewer.
    Audit the following implementation against the system architecture and requirements.

    [Architecture Specification]
    {state['arch_text'][:3000]}

    [Requirements Specification]
    {state['requirements_text'][:2000]}

    [Source Codebase]
    {current_code}

    Audit Checklist:
    1. SECURITY (OWASP Top 10): Check for SQL/Command injection, XSS, insecure deserialization, missing auth/RBAC checks, hardcoded API keys/secrets, and unsafe path traversals.
    2. PERFORMANCE: Flag N+1 queries, unbounded loops, blocking I/O in async paths, or missing resource cleanup.
    3. ARCHITECTURE ALIGNMENT: Verify code matches the contracts, data models, and error handling defined in `architecture.md`.
    4. CODE QUALITY: Check for missing type hints, unhandled exceptions, dead code, and poor modularity.

    Rules:
    - Set `is_approved = False` if ANY 'CRITICAL' or 'HIGH' severity issues exist.
    - Set `is_approved = True` if only 'MEDIUM' or 'LOW' recommendations remain.
    - Generate a comprehensive `audit_markdown` report suitable for saving as `docs/SECURITY_AUDIT.md`.
    """
    
    report = llm.with_structured_output(ReviewAuditPayload).invoke(prompt)
    new_iter = state.get("iteration_count", 0) + 1
    
    sys.stderr.write(
        f"Audit Pass #{new_iter}: {'APPROVED' if report.is_approved else 'REJECTED'} "
        f"({len(report.findings)} findings)\n"
    )
    return {
        "codebase_snapshot": current_code,
        "audit_report": report,
        "iteration_count": new_iter
    }


def code_remediation_engineer(state: ReviewState):
    sys.stderr.write("\n[Phase 3 -> Remediator] Patching security and quality blockers...\n")
    report = state["audit_report"]
    
    blockers = [f for f in report.findings if f.severity in ("CRITICAL", "HIGH")]
    findings_text = "\n".join(
        f"- [{f.severity}] ({f.category}) in `{f.file_path}` ({f.line_hint}): {f.description}\n  Fix: {f.remediation_plan}"
        for f in blockers
    )

    prompt = f"""
    You are a Senior Security & Refactoring Engineer.
    The code review audit rejected the current implementation due to the following blockers:

    {findings_text}

    Current Codebase:
    {state['codebase_snapshot']}

    Task:
    Rewrite the affected files to resolve every CRITICAL and HIGH severity finding cleanly 
    without breaking existing functionality or public method signatures.
    Return the full, complete file contents for every modified file.
    """
    
    remediation = llm.with_structured_output(RemediationOutput).invoke(prompt)
    workspace = Path(state["workspace_path"])
    
    for patch in remediation.patched_files:
        target = workspace / patch.file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patch.updated_content, encoding="utf-8")
        sys.stderr.write(f"Patched file: {patch.file_path}\n")
        
    return state


def audit_report_writer(state: ReviewState):
    sys.stderr.write("\n[Phase 3 -> Writer] Saving docs/SECURITY_AUDIT.md...\n")
    docs_dir = Path(state["workspace_path"]) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = docs_dir / "SECURITY_AUDIT.md"
    report_path.write_text(state["audit_report"].audit_markdown, encoding="utf-8")
    sys.stderr.write(f"Audit report persisted to: {report_path.resolve()}\n")
    return state


# =====================================================================
# 5. CONDITIONAL ROUTING & GRAPH COMPILATION
# =====================================================================

def route_after_review(state: ReviewState):
    report = state.get("audit_report")
    iterations = state.get("iteration_count", 0)
    
    if report and report.is_approved:
        return "approved"
    
    if iterations >= 3:
        sys.stderr.write(f"Max remediation cycles ({iterations}) reached. Writing final audit report.\n")
        return "approved"
        
    return "remediate"


workflow = StateGraph(ReviewState)

workflow.add_node("auditor", appsec_and_quality_auditor)
workflow.add_node("remediator", code_remediation_engineer)
workflow.add_node("writer", audit_report_writer)

workflow.add_edge(START, "auditor")
workflow.add_conditional_edges(
    "auditor",
    route_after_review,
    {
        "approved": "writer",
        "remediate": "remediator"
    }
)
workflow.add_edge("remediator", "auditor")
workflow.add_edge("writer", END)

review_orchestrator = workflow.compile()


# =====================================================================
# 6. MCP TOOL REGISTRATION & SERVER ENTRY POINT
# =====================================================================

@mcp.tool()
def review_and_harden_codebase(
    target_dir: str = "src",
    workspace_path: str = "."
) -> str:
    """
    Executes Phase 3 of SDLC: Performs an AppSec (OWASP/Secrets), Performance, and Code Quality 
    audit on the codebase, iteratively patches CRITICAL/HIGH findings, and generates docs/SECURITY_AUDIT.md.
    
    Args:
        target_dir: Subdirectory containing source code to review (e.g., 'src' or '.').
        workspace_path: Root repository directory.
    """
    root = Path(workspace_path)
    docs_dir = root / "docs"
    
    arch_file = docs_dir / "architecture.md"
    req_file = docs_dir / "requirements.md"

    initial_state: ReviewState = {
        "workspace_path": str(root.resolve()),
        "target_dir": target_dir,
        "arch_text": arch_file.read_text(encoding="utf-8") if arch_file.exists() else "",
        "requirements_text": req_file.read_text(encoding="utf-8") if req_file.exists() else "",
        "codebase_snapshot": "",
        "audit_report": None,
        "iteration_count": 0
    }
    
    final_state = review_orchestrator.invoke(initial_state)
    report = final_state.get("audit_report")
    
    status = "PASSED" if (report and report.is_approved) else "COMPLETED WITH WARNINGS"
    finding_count = len(report.findings) if report else 0
    
    return (
        f"Phase 3 Code Review & AppSec Audit {status} ({finding_count} total findings logged). "
        f"Full report saved to {workspace_path}/docs/SECURITY_AUDIT.md."
    )


if __name__ == "__main__":
    mcp.run()