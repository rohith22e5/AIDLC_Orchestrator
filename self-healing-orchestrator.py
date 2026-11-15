import ast
import os
import re
import sys
import subprocess
from datetime import datetime
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

mcp = FastMCP("SDLC-Observability-SelfHealing-Orchestrator")

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0)
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)


# =====================================================================
# 2. STRUCTURED SCHEMAS
# =====================================================================

class IncidentDiagnosis(BaseModel):
    incident_title: str = Field(description="Concise SRE incident title")
    severity: str = Field(description="'SEV-1', 'SEV-2', or 'SEV-3'")
    affected_files: List[str] = Field(description="Relative paths of source files causing the failure")
    root_cause_analysis: str = Field(description="Deep technical explanation of why the runtime failure occurred")
    reproduction_steps: str = Field(description="Exact inputs/state required to reproduce the bug in a unit test")
    fix_strategy: str = Field(description="Code-level remediation plan")

class CodeArtifact(BaseModel):
    file_path: str = Field(description="Relative file path e.g., src/api/routes.py or tests/regression/test_inc_01.py")
    content: str = Field(description="Complete, valid file content")

class SelfHealingState(TypedDict):
    workspace_path: str
    incident_logs: str
    test_runner_command: str
    codebase_snapshot: str
    arch_text: str
    
    diagnosis: Optional[IncidentDiagnosis]
    written_artifacts: List[CodeArtifact]
    pr_summary: str
    
    last_test_stdout: str
    last_test_stderr: str
    last_test_exit_code: int
    
    iteration_count: int


# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================

def extract_and_write_artifacts(workspace: Path, llm_text: str) -> List[CodeArtifact]:
    """Parses ```lang blocks preceded by 'FILE: path/to/file' and writes clean multi-line files."""
    pattern = r"FILE:\s*([^\n`]+)\s*\n```[a-zA-Z0-9_-]*\n(.*?)```"
    matches = re.findall(pattern, llm_text, flags=re.DOTALL)
    written: List[CodeArtifact] = []

    for rel_path, code in matches:
        clean_path = rel_path.strip().strip("`*")
        target = workspace / clean_path
        target.parent.mkdir(parents=True, exist_ok=True)
        clean_code = code.strip() + "\n"
        target.write_text(clean_code, encoding="utf-8")
        sys.stderr.write(f"Wrote file: {clean_path}\n")
        written.append(CodeArtifact(file_path=clean_path, content=clean_code))

    if not written:
        sys.stderr.write("Warning: No 'FILE: <path>' code blocks matched in LLM response.\n")
    return written

def read_codebase(workspace: Path) -> str:
    ignore_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", "docs", ".pytest_cache"}
    allowed_exts = {".py", ".ts", ".js", ".go", ".sql", ".yml", ".yaml"}
    chunks = []
    for path in sorted(workspace.rglob("*")):
        if any(part in ignore_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix in allowed_exts and "test" not in path.parts:
            rel = path.relative_to(workspace).as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks.append(f"### FILE: {rel}\n{text[:3000]}")
    return "\n\n".join(chunks)[:12000]

def get_existing_tests(workspace: Path) -> str:
    """Collects existing unit and integration tests so hotfixes preserve backward compatibility."""
    tests_dir = workspace / "tests"
    if not tests_dir.exists():
        return ""
    chunks = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        if "regression" not in path.parts:
            rel = path.relative_to(workspace).as_posix()
            chunks.append(f"### {rel}\n{path.read_text(encoding='utf-8', errors='ignore')[:1500]}")
    return "\n\n".join(chunks)[:3500]


# =====================================================================
# 4. AGENT NODES
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
    3. Determine the root cause (e.g., unhandled null/None, race condition, timeout, schema mismatch, memory leak).
    4. Define exact reproduction inputs so an engineer can write a failing regression test.
    """
    
    diagnosis = llm.with_structured_output(IncidentDiagnosis).invoke(prompt)
    sys.stderr.write(f"Incident Triaged: [{diagnosis.severity}] {diagnosis.incident_title}\n")
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
       for out-of-range values like `-5`) rather than brittle exact exception message strings.
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
    written = extract_and_write_artifacts(workspace, response.content)
    updated_snapshot = read_codebase(workspace)
    
    return {
        "written_artifacts": written,
        "codebase_snapshot": updated_snapshot,
        "pr_summary": f"Hotfix [{diag.severity}]: {diag.incident_title} - {diag.fix_strategy}"
    }


def regression_verification_runner(state: SelfHealingState):
    cmd = state.get("test_runner_command") or "pytest -v"
    if "pytest" in cmd and "--maxfail" not in cmd:
        cmd = f"{cmd} --maxfail=1"
    sys.stderr.write(f"\n[Phase 6 -> Runner] Executing full regression & unit test suite: `{cmd}`...\n")
    
    workspace = Path(state["workspace_path"])
    new_iter = state.get("iteration_count", 0) + 1

    # Fast AST syntax validation before invoking pytest
    for f in state.get("written_artifacts", []):
        if f.file_path.endswith(".py"):
            try:
                ast.parse(f.content)
            except SyntaxError as e:
                err_msg = f"SyntaxError in {f.file_path} at line {e.lineno}: {e.msg}"
                sys.stderr.write(f"{err_msg}\n")
                return {
                    "last_test_stdout": "",
                    "last_test_stderr": err_msg,
                    "last_test_exit_code": 2,
                    "iteration_count": new_iter
                }

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workspace.resolve()}{os.pathsep}{(workspace / 'src').resolve()}"

    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60
        )
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except Exception as e:
        stdout, stderr, code = "", str(e), 1

    sys.stderr.write(f"Verification Pass #{new_iter} Exit Code: {code}\n")
    return {
        "last_test_stdout": stdout,
        "last_test_stderr": stderr,
        "last_test_exit_code": code,
        "iteration_count": new_iter
    }


def observability_and_postmortem_agent(state: SelfHealingState):
    sys.stderr.write("\n[Phase 6 -> Agent 3] Hardening observability alerts & generating Post-Mortem...\n")
    diag = state["diagnosis"]
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
    workspace = Path(state["workspace_path"])
    extract_and_write_artifacts(workspace, alerts_res.content)

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
    sys.stderr.write(f"Post-mortem saved to: {(docs_dir / 'POSTMORTEM.md').resolve()}\n")
    
    return state


# =====================================================================
# 5. CONDITIONAL ROUTING & GRAPH COMPILATION
# =====================================================================

def route_hotfix_verification(state: SelfHealingState):
    if state.get("last_test_exit_code", 1) == 0:
        return "verified"
    if state.get("iteration_count", 0) >= 3:
        sys.stderr.write("Max self-healing repair attempts (3) reached. Proceeding to post-mortem.\n")
        return "verified"
    return "retry_hotfix"


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
    {
        "verified": "postmortem_writer",
        "retry_hotfix": "hotfix_engineer"
    }
)
workflow.add_edge("postmortem_writer", END)

self_healing_orchestrator = workflow.compile()


# =====================================================================
# 6. MCP TOOL REGISTRATION & SERVER ENTRY POINT
# =====================================================================

@mcp.tool()
def heal_runtime_incident(
    incident_logs: str,
    test_runner_command: str = "pytest -v",
    workspace_path: str = "."
) -> str:
    """
    Executes Phase 6 of SDLC: Triages production stack traces or telemetry alerts, writes an 
    automated regression test, patches the source code, verifies the fix against the test suite, 
    adds Prometheus/OTel alert rules, and writes docs/POSTMORTEM.md.
    
    Args:
        incident_logs: Raw stack trace, error log snippet, or monitoring alert payload.
        test_runner_command: Command to run unit and regression tests (e.g., 'pytest -v').
        workspace_path: Root repository directory.
    """
    root = Path(workspace_path).resolve()
    arch_file = root / "docs" / "architecture.md"

    initial_state: SelfHealingState = {
        "workspace_path": str(root),
        "incident_logs": incident_logs.strip(),
        "test_runner_command": test_runner_command,
        "codebase_snapshot": read_codebase(root),
        "arch_text": arch_file.read_text(encoding="utf-8") if arch_file.exists() else "",
        "diagnosis": None,
        "written_artifacts": [],
        "pr_summary": "",
        "last_test_stdout": "",
        "last_test_stderr": "",
        "last_test_exit_code": 1,
        "iteration_count": 0
    }
    
    final_state = self_healing_orchestrator.invoke(initial_state)
    diag = final_state.get("diagnosis")
    passed = final_state.get("last_test_exit_code", 1) == 0
    
    status = "RESOLVED & VERIFIED" if passed else "PATCHED (REQUIRES MANUAL REVIEW)"
    return (
        f"Phase 6 Self-Healing {status} for [{diag.severity if diag else 'INCIDENT'}] "
        f"{diag.incident_title if diag else ''}. "
        f"Alert rules updated under monitoring/ and post-mortem saved to {workspace_path}/docs/POSTMORTEM.md."
    )


if __name__ == "__main__":
    mcp.run()