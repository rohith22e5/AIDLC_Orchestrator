import os
import sys
import subprocess
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

mcp = FastMCP("SDLC-DevOps-IaC-Orchestrator")

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0)
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)


# =====================================================================
# 2. STRUCTURED SCHEMAS
# =====================================================================

class InfraArtifact(BaseModel):
    file_path: str = Field(description="Relative path e.g., Dockerfile, .github/workflows/ci-cd.yml, or infra/main.tf")
    content: str = Field(description="Complete, production-ready file content")

class CICDPayload(BaseModel):
    artifacts: List[InfraArtifact] = Field(
        description="Includes multi-stage Dockerfile, .dockerignore, and .github/workflows/ci-cd.yml"
    )
    exposed_port: int = Field(description="Primary application container port")

class IaCPayload(BaseModel):
    artifacts: List[InfraArtifact] = Field(
        description="Declarative IaC manifests under infra/ (e.g., main.tf, variables.tf, outputs.tf or k8s deployment/service)"
    )
    target_provider: str = Field(description="Target deployment platform e.g., AWS ECS/Fargate, GCP Cloud Run, or Kubernetes")

class DevSecOpsFinding(BaseModel):
    severity: str = Field(description="'BLOCKER' or 'WARNING'")
    target_agent: str = Field(description="'cicd_engineer' or 'iac_engineer'")
    file_path: str
    issue: str
    remediation: str

class DevSecOpsAuditPayload(BaseModel):
    is_approved: bool = Field(description="True ONLY if zero BLOCKER findings exist")
    findings: List[DevSecOpsFinding] = Field(default_factory=list)
    deployment_runbook_md: str = Field(
        description="Complete docs/DEPLOYMENT_GUIDE.md covering build, secret configuration, CI/CD triggers, and rollback steps"
    )

class DevOpsState(TypedDict):
    workspace_path: str
    cloud_target: str
    iac_tool: str
    arch_text: str
    requirements_text: str
    codebase_summary: str
    
    cicd_artifacts: List[InfraArtifact]
    iac_artifacts: List[InfraArtifact]
    
    audit: Optional[DevSecOpsAuditPayload]
    iteration_count: int


# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================

def write_artifacts(workspace: Path, artifacts: List[InfraArtifact]):
    for item in artifacts:
        target = workspace / item.file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.content, encoding="utf-8")
        sys.stderr.write(f"Wrote DevOps artifact: {item.file_path}\n")

def inspect_project_manifest(workspace: Path) -> str:
    """Collects dependency manifests and entrypoints to guide Dockerfile & CI generation."""
    manifest_names = {
        "pyproject.toml", "requirements.txt", "uv.lock",
        "package.json", "go.mod", "Cargo.toml", "docker-compose.test.yml"
    }
    ignore_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", "docs"}
    snippets = []

    for path in sorted(workspace.rglob("*")):
        if any(part in ignore_dirs for part in path.parts):
            continue
        if path.is_file() and (path.name in manifest_names or "src" in path.parts):
            rel = path.relative_to(workspace).as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")
            snippets.append(f"--- {rel} ---\n{text[:2000]}")
    return "\n\n".join(snippets)[:10000]


# =====================================================================
# 4. AGENT NODES
# =====================================================================

def cicd_and_container_engineer(state: DevOpsState):
    sys.stderr.write("\n[Phase 5 -> Agent 1] Generating production Dockerfile & GitHub Actions CI/CD...\n")
    
    critique_context = ""
    if state.get("audit") and not state["audit"].is_approved:
        cicd_issues = [f for f in state["audit"].findings if f.target_agent == "cicd_engineer"]
        if cicd_issues:
            critique_context = "\nDEVSECOPS AUDIT FIXES REQUIRED:\n" + "\n".join(
                f"- {i.file_path}: {i.issue} -> {i.remediation}" for i in cicd_issues
            )

    prompt = f"""
    You are a Staff DevOps & Release Engineer.
    Generate production-hardened containerization and CI/CD pipeline files for this project.

    [Architecture Specification]
    {state['arch_text'][:3000]}

    [Project Files & Dependency Manifests]
    {state['codebase_summary'][:5000]}
    {critique_context}

    Mandatory Deliverables:
    1. `Dockerfile`: Multi-stage production build using slim/distroless base images, pinned base tags, 
       a dedicated non-root user (`USER appuser`), and a `HEALTHCHECK` instruction.
    2. `.dockerignore`: Exclude `.git`, `.venv`, `__pycache__`, `.env*`, `tests/`, and `docs/`.
    3. `.github/workflows/ci-cd.yml`: Complete GitHub Actions workflow triggered on push/PR to `main`:
       - Job 1: Lint & Unit/Integration Tests
       - Job 2: Container Build & Trivy Vulnerability Scan
       - Job 3: Deploy stage targeting {state['cloud_target']} using OIDC/secrets (no hardcoded credentials).
    """
    
    payload = llm.with_structured_output(CICDPayload).invoke(prompt)
    write_artifacts(Path(state["workspace_path"]), payload.artifacts)
    return {"cicd_artifacts": payload.artifacts}


def cloud_infrastructure_engineer(state: DevOpsState):
    sys.stderr.write(f"\n[Phase 5 -> Agent 2] Generating {state['iac_tool']} IaC for {state['cloud_target']}...\n")
    
    critique_context = ""
    if state.get("audit") and not state["audit"].is_approved:
        iac_issues = [f for f in state["audit"].findings if f.target_agent == "iac_engineer"]
        if iac_issues:
            critique_context = "\nDEVSECOPS AUDIT FIXES REQUIRED:\n" + "\n".join(
                f"- {i.file_path}: {i.issue} -> {i.remediation}" for i in iac_issues
            )

    prompt = f"""
    You are a Principal Cloud Infrastructure Architect.
    Generate declarative Infrastructure as Code (`{state['iac_tool']}`) targeting `{state['cloud_target']}` under `infra/`.

    [Architecture & Backing Services]
    {state['arch_text'][:3500]}

    [Non-Functional Requirements]
    {state['requirements_text'][:2000]}

    [Generated CI/CD & Container Config]
    {[a.file_path for a in state.get('cicd_artifacts', [])]}
    {critique_context}

    Mandatory Deliverables:
    1. Provision compute (containers/serverless/cluster), managed database/cache resources referenced in architecture, 
       and least-privilege IAM roles.
    2. Enforce encryption at rest, private networking/security groups, and secret manager references (never plaintext secrets).
    3. Structure cleanly under `infra/` (e.g., `infra/main.tf`, `infra/variables.tf`, `infra/outputs.tf` for Terraform, 
       or `infra/k8s/deployment.yaml`, `infra/k8s/service.yaml`, `infra/k8s/hpa.yaml` for Kubernetes).
    """
    
    payload = llm.with_structured_output(IaCPayload).invoke(prompt)
    write_artifacts(Path(state["workspace_path"]), payload.artifacts)
    return {"iac_artifacts": payload.artifacts}


def devsecops_auditor(state: DevOpsState):
    sys.stderr.write("\n[Phase 5 -> Quality Gate] DevSecOps Auditor verifying Docker, CI/CD & IaC security...\n")
    
    cicd_dump = "\n\n".join(f"### {a.file_path}\n{a.content}" for a in state.get("cicd_artifacts", []))
    iac_dump = "\n\n".join(f"### {a.file_path}\n{a.content}" for a in state.get("iac_artifacts", []))

    # Optional deterministic validation if terraform/docker CLI is available locally
    cli_checks = []
    workspace = Path(state["workspace_path"])
    if (workspace / "infra").exists() and state["iac_tool"].lower() == "terraform":
        try:
            proc = subprocess.run(
                ["terraform", "fmt", "-check", "-recursive"],
                cwd=workspace / "infra",
                capture_output=True,
                text=True,
                timeout=15
            )
            cli_checks.append(f"terraform fmt exit={proc.returncode}: {proc.stderr or 'OK'}")
        except FileNotFoundError:
            cli_checks.append("terraform CLI not installed locally; relying on static AST/LLM audit.")

    prompt = f"""
    You are a Staff DevSecOps & Cloud Security Auditor.
    Audit all generated container, CI/CD pipeline, and Infrastructure as Code files.

    [Container & CI/CD Files]
    {cicd_dump}

    [Infrastructure as Code Files]
    {iac_dump}

    [Local CLI Verification Output]
    {chr(10).join(cli_checks) if cli_checks else 'None'}

    Security & Reliability Checklist:
    1. Container Hardening: Must NOT run as `root` in the final stage; must use multi-stage build and `.dockerignore`.
    2. Secret Hygiene: Zero hardcoded API keys, passwords, or tokens in YAML, Dockerfile, or IaC files.
    3. CI/CD Gates: Workflow must run automated tests and vulnerability scanning before deployment.
    4. Cloud Security: Security groups/firewalls must restrict ingress; databases must not be publicly exposed; IAM must follow least privilege.
    5. Syntax Validity: YAML, Dockerfile, and HCL/Kubernetes manifests must be syntactically valid and consistent on ports/env vars.

    Rules:
    - Set `is_approved = False` if ANY 'BLOCKER' issue is found, and assign `target_agent` ('cicd_engineer' or 'iac_engineer').
    - Set `is_approved = True` once hardened and valid.
    - Always generate a thorough `deployment_runbook_md` for `docs/DEPLOYMENT_GUIDE.md`.
    """
    
    audit = llm.with_structured_output(DevSecOpsAuditPayload).invoke(prompt)
    new_iter = state.get("iteration_count", 0) + 1
    sys.stderr.write(
        f"DevSecOps Audit Pass #{new_iter}: {'APPROVED' if audit.is_approved else 'REJECTED'} "
        f"({len(audit.findings)} findings)\n"
    )
    return {"audit": audit, "iteration_count": new_iter}


def deployment_guide_writer(state: DevOpsState):
    sys.stderr.write("\n[Phase 5 -> Writer] Writing docs/DEPLOYMENT_GUIDE.md...\n")
    docs_dir = Path(state["workspace_path"]) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    guide_path = docs_dir / "DEPLOYMENT_GUIDE.md"
    guide_path.write_text(state["audit"].deployment_runbook_md, encoding="utf-8")
    sys.stderr.write(f"Deployment guide persisted to: {guide_path.resolve()}\n")
    return state


# =====================================================================
# 5. CONDITIONAL ROUTING & GRAPH COMPILATION
# =====================================================================

def route_devsecops_audit(state: DevOpsState):
    audit = state.get("audit")
    iterations = state.get("iteration_count", 0)
    
    if audit and audit.is_approved:
        return "approved"
    
    if iterations >= 3:
        sys.stderr.write(f"Max DevSecOps audit cycles ({iterations}) reached. Writing final deployment guide.\n")
        return "approved"
        
    targets = {f.target_agent for f in audit.findings if f.severity == "BLOCKER"}
    if "cicd_engineer" in targets:
        return "fix_cicd"
    return "fix_iac"


workflow = StateGraph(DevOpsState)

workflow.add_node("cicd_engineer", cicd_and_container_engineer)
workflow.add_node("iac_engineer", cloud_infrastructure_engineer)
workflow.add_node("auditor", devsecops_auditor)
workflow.add_node("writer", deployment_guide_writer)

workflow.add_edge(START, "cicd_engineer")
workflow.add_edge("cicd_engineer", "iac_engineer")
workflow.add_edge("iac_engineer", "auditor")

workflow.add_conditional_edges(
    "auditor",
    route_devsecops_audit,
    {
        "approved": "writer",
        "fix_cicd": "cicd_engineer",
        "fix_iac": "iac_engineer",
    }
)
workflow.add_edge("writer", END)

devops_orchestrator = workflow.compile()


# =====================================================================
# 6. MCP TOOL REGISTRATION & SERVER ENTRY POINT
# =====================================================================

@mcp.tool()
def generate_cicd_and_infrastructure(
    cloud_target: str = "AWS ECS Fargate",
    iac_tool: str = "Terraform",
    workspace_path: str = "."
) -> str:
    """
    Executes Phase 5 of SDLC: Generates a hardened multi-stage Dockerfile, GitHub Actions CI/CD 
    workflow, and declarative Cloud IaC (Terraform/Kubernetes), audits them via DevSecOps checks, 
    and writes docs/DEPLOYMENT_GUIDE.md.
    
    Args:
        cloud_target: Target cloud runtime (e.g., 'AWS ECS Fargate', 'GCP Cloud Run', 'Kubernetes').
        iac_tool: IaC framework to generate (e.g., 'Terraform', 'Kubernetes', 'OpenTofu').
        workspace_path: Root repository directory containing /docs and /src.
    """
    root = Path(workspace_path).resolve()
    docs_dir = root / "docs"
    
    arch_file = docs_dir / "architecture.md"
    req_file = docs_dir / "requirements.md"
    
    if not arch_file.exists():
        return "Error: docs/architecture.md not found. Run Phase 1 before Phase 5."

    initial_state: DevOpsState = {
        "workspace_path": str(root),
        "cloud_target": cloud_target,
        "iac_tool": iac_tool,
        "arch_text": arch_file.read_text(encoding="utf-8"),
        "requirements_text": req_file.read_text(encoding="utf-8") if req_file.exists() else "",
        "codebase_summary": inspect_project_manifest(root),
        "cicd_artifacts": [],
        "iac_artifacts": [],
        "audit": None,
        "iteration_count": 0
    }
    
    final_state = devops_orchestrator.invoke(initial_state)
    audit = final_state.get("audit")
    
    status = "APPROVED" if (audit and audit.is_approved) else "COMPLETED WITH WARNINGS"
    return (
        f"Phase 5 CI/CD & IaC Generation {status} ({iac_tool} for {cloud_target}). "
        f"Runbook saved to {workspace_path}/docs/DEPLOYMENT_GUIDE.md."
    )


if __name__ == "__main__":
    mcp.run()