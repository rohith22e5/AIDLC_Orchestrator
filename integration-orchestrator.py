import ast
import os
import re
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

mcp = FastMCP("SDLC-Integration-E2E-Orchestrator")

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0)
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)


# =====================================================================
# 2. STRUCTURED SCHEMAS
# =====================================================================

class GeneratedArtifact(BaseModel):
    file_path: str = Field(description="Relative path e.g., docker-compose.test.yml or tests/integration/test_api_e2e.py")
    content: str = Field(description="Complete, valid file content")

class IntegrationDiagnostic(BaseModel):
    passed: bool
    failure_layer: str = Field(description="'NONE', 'INFRASTRUCTURE_CONFIG', 'TEST_SUITE', or 'APP_SOURCE'")
    root_cause: str = Field(description="Exact analysis of connection errors, schema mismatches, or HTTP status failures")
    fix_instructions: str = Field(description="Actionable instructions to fix compose config, tests, or source code")

class IntegrationState(TypedDict):
    workspace_path: str
    arch_text: str
    requirements_text: str
    codebase_summary: str
    
    env_artifacts: List[GeneratedArtifact]
    test_artifacts: List[GeneratedArtifact]
    integration_test_cmd: str
    
    last_stdout: str
    last_stderr: str
    container_logs: str
    last_exit_code: int
    
    diagnostic: Optional[IntegrationDiagnostic]
    report_markdown: str
    iteration_count: int


# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================

def extract_and_write_artifacts(workspace: Path, llm_text: str) -> List[GeneratedArtifact]:
    """Parses ```lang blocks preceded by 'FILE: path/to/file' and writes clean multi-line files."""
    pattern = r"FILE:\s*([^\n`]+)\s*\n```[a-zA-Z0-9_-]*\n(.*?)```"
    matches = re.findall(pattern, llm_text, flags=re.DOTALL)
    written: List[GeneratedArtifact] = []

    for rel_path, code in matches:
        clean_path = rel_path.strip().strip("`*")
        target = workspace / clean_path
        target.parent.mkdir(parents=True, exist_ok=True)
        clean_code = code.strip() + "\n"
        target.write_text(clean_code, encoding="utf-8")
        sys.stderr.write(f"Wrote artifact: {clean_path}\n")
        written.append(GeneratedArtifact(file_path=clean_path, content=clean_code))

    if not written:
        sys.stderr.write("Warning: No 'FILE: <path>' code blocks matched in LLM response.\n")
    return written

def get_source_overview(workspace: Path) -> str:
    ignore_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", "docs", ".pytest_cache"}
    allowed_exts = {".py", ".ts", ".js", ".go", ".sql", ".json", ".toml", ".yml", ".yaml"}
    chunks = []
    for path in sorted(workspace.rglob("*")):
        if any(part in ignore_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix in allowed_exts and "test" not in path.parts:
            rel = path.relative_to(workspace).as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks.append(f"--- {rel} ---\n{text[:2500]}")
    return "\n\n".join(chunks)[:12000]


# =====================================================================
# 4. AGENT NODES
# =====================================================================

def environment_provisioner(state: IntegrationState):
    sys.stderr.write("\n[Phase 4 -> Agent 1] Environment Provisioner generating Docker Compose stack...\n")
    
    fix_context = ""
    if state.get("diagnostic") and state["diagnostic"].failure_layer == "INFRASTRUCTURE_CONFIG":
        fix_context = f"\nINFRASTRUCTURE FIX REQUIRED:\n{state['diagnostic'].fix_instructions}"

    fence = "`" * 3
    prompt = f"""
    You are a Senior DevOps & Test Environment Engineer.
    Based on the architecture and source code below, generate an isolated Docker Compose test environment 
    (`docker-compose.test.yml` and `.env.test`) to run live integration and end-to-end tests.

    [Architecture]
    {state['arch_text'][:3500]}

    [Source Code Overview]
    {state['codebase_summary'][:5000]}
    {fix_context}

    Requirements:
    1. Define any required backing services (e.g., PostgreSQL, Redis, MongoDB, RabbitMQ) with healthchecks.
       If the application is pure in-memory Python with no external services required, generate a minimal valid compose file.
    2. Use non-conflicting host ports or internal Docker bridge networking so it runs cleanly locally.
    3. Include `.env.test` with connection strings matching the service hostnames/ports.

    Output EVERY file strictly using this Markdown format (do NOT output JSON):

    FILE: docker-compose.test.yml
    {fence}yaml
    # complete docker-compose config here
    {fence}

    FILE: .env.test
    {fence}env
    # environment variables here
    {fence}
    """
    
    response = llm.invoke(prompt)
    artifacts = extract_and_write_artifacts(Path(state["workspace_path"]), response.content)
    return {"env_artifacts": artifacts}


def integration_test_engineer(state: IntegrationState):
    sys.stderr.write("\n[Phase 4 -> Agent 2] E2E Test Engineer writing multi-service & API contract tests...\n")
    
    fix_context = ""
    if state.get("diagnostic") and state["diagnostic"].failure_layer == "TEST_SUITE":
        fix_context = (
            f"\nTEST SUITE FIX REQUIRED:\n"
            f"Root Cause: {state['diagnostic'].root_cause}\n"
            f"Fix Instructions: {state['diagnostic'].fix_instructions}\n"
            f"Last Error Output:\n{(state['last_stderr'] or state['last_stdout'])[-1500:]}\n"
        )

    fence = "`" * 3
    prompt = f"""
    You are a Staff QA Automation & Integration Test Engineer.
    Author a comprehensive integration & E2E test suite under `tests/integration/test_e2e_workflows.py`.

    [Requirements]
    {state['requirements_text'][:2500]}

    [Architecture & API Contracts]
    {state['arch_text'][:3500]}

    [Source Code Overview]
    {state['codebase_summary'][:4500]}
    {fix_context}

    Mandates:
    1. Test real end-to-end user flows across API endpoints, database persistence, and cache/queue interactions.
    2. Import application modules directly from `src/` (e.g., `from cache import TTLCache`).
    3. Do NOT mock internal databases or caches—connect to the live backing services defined in `.env.test`.
    4. Keep any sleep/TTL expiration timers fast (< 0.2s) and include teardown/cleanup fixtures.

    Output EVERY file strictly using this Markdown format (do NOT output JSON):

    FILE: tests/integration/test_e2e_workflows.py
    {fence}python
    # complete multi-line integration test suite here
    {fence}
    """
    
    response = llm.invoke(prompt)
    test_files = extract_and_write_artifacts(Path(state["workspace_path"]), response.content)
    return {
        "test_artifacts": test_files,
        "integration_test_cmd": state.get("integration_test_cmd") or "pytest tests/integration -v"
    }


def integration_stack_runner(state: IntegrationState):
    sys.stderr.write("\n[Phase 4 -> Runner] Booting Docker Compose services & executing E2E suite...\n")
    workspace = Path(state["workspace_path"])
    compose_file = workspace / "docker-compose.test.yml"
    test_cmd = state.get("integration_test_cmd") or "pytest tests/integration -v"
    if "pytest" in test_cmd and "--maxfail" not in test_cmd:
        test_cmd = f"{test_cmd} --maxfail=1"
    
    # Fast AST syntax validation before booting containers or running pytest
    for f in state.get("test_artifacts", []):
        if f.file_path.endswith(".py"):
            try:
                ast.parse(f.content)
            except SyntaxError as e:
                err_msg = f"SyntaxError in {f.file_path} at line {e.lineno}: {e.msg}"
                sys.stderr.write(f"{err_msg}\n")
                return {
                    "last_stdout": "",
                    "last_stderr": err_msg,
                    "container_logs": "",
                    "last_exit_code": 1
                }

    stdout, stderr, container_logs = "", "", ""
    exit_code = 1

    # Export both workspace root and src/ so both `import src.module` and `import module` resolve
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workspace.resolve()}{os.pathsep}{(workspace / 'src').resolve()}"

    try:
        if compose_file.exists():
            sys.stderr.write("Starting backing containers via docker compose...\n")
            up_proc = subprocess.run(
                ["docker", "compose", "-f", "docker-compose.test.yml", "up", "-d", "--wait"],
                cwd=workspace,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=120
            )
            if up_proc.returncode != 0:
                stderr += f"\n[Docker Compose Up Failed]\n{up_proc.stderr}"
                raise RuntimeError("Docker Compose failed to start services.")

        sys.stderr.write(f"Running integration command: {test_cmd}\n")
        test_proc = subprocess.run(
            test_cmd,
            shell=True,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120
        )
        stdout = test_proc.stdout
        stderr += test_proc.stderr
        exit_code = test_proc.returncode

        if exit_code != 0 and compose_file.exists():
            logs_proc = subprocess.run(
                ["docker", "compose", "-f", "docker-compose.test.yml", "logs", "--tail=100"],
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30
            )
            container_logs = logs_proc.stdout

    except Exception as e:
        stderr += f"\n[Execution Exception]: {str(e)}"
        exit_code = 1

    finally:
        if compose_file.exists():
            sys.stderr.write("Tearing down test containers...\n")
            subprocess.run(
                ["docker", "compose", "-f", "docker-compose.test.yml", "down", "-v"],
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=60
            )

    sys.stderr.write(f"Integration run completed with exit code: {exit_code}\n")
    return {
        "last_stdout": stdout,
        "last_stderr": stderr,
        "container_logs": container_logs,
        "last_exit_code": exit_code
    }


def integration_debugger(state: IntegrationState):
    sys.stderr.write("\n[Phase 4 -> Debugger] Analyzing E2E test & container telemetry...\n")
    exit_code = state["last_exit_code"]
    
    if exit_code == 0:
        sys.stderr.write("Integration & E2E verification PASSED!\n")
        diag = IntegrationDiagnostic(
            passed=True,
            failure_layer="NONE",
            root_cause="All multi-service and end-to-end API contract tests passed.",
            fix_instructions="None"
        )
    else:
        sys.stderr.write("Integration verification FAILED. Diagnosing root cause...\n")
        prompt = f"""
        You are a Principal Distributed Systems & QA Debugger.
        Integration tests failed with exit code {exit_code}.

        [Test Stdout]
        {state['last_stdout'][-2500:]}

        [Test Stderr]
        {state['last_stderr'][-2500:]}

        [Docker Container Logs]
        {state['container_logs'][-2000:]}

        Classify the failure layer strictly as one of:
        - 'INFRASTRUCTURE_CONFIG' (port conflicts, connection refused, bad docker-compose healthcheck/env)
        - 'TEST_SUITE' (wrong endpoint URL, wrong import path, bad fixture setup, test assertion bug)
        - 'APP_SOURCE' (unhandled 500 error, SQL query failure, serialization bug in src/)

        Provide exact root cause and fix instructions.
        """
        diag = llm.with_structured_output(IntegrationDiagnostic).invoke(prompt)
        sys.stderr.write(f"Classified Failure Layer: {diag.failure_layer} -> {diag.root_cause}\n")

    new_iter = state.get("iteration_count", 0) + 1
    return {"diagnostic": diag, "iteration_count": new_iter}


def app_hotfix_engineer(state: IntegrationState):
    sys.stderr.write("\n[Phase 4 -> Hotfix Dev] Patching application source code for integration issue...\n")
    diag = state["diagnostic"]
    
    fence = "`" * 3
    prompt = f"""
    You are a Senior Backend Engineer. Integration testing uncovered a bug in the application source code.
    
    Root Cause: {diag.root_cause}
    Fix Instructions: {diag.fix_instructions}

    Current Source Code:
    {state['codebase_summary']}

    Output EVERY updated source file under `src/` strictly using this Markdown format (do NOT output JSON):

    FILE: src/example.py
    {fence}python
    # complete patched source code here
    {fence}
    """
    response = llm.invoke(prompt)
    extract_and_write_artifacts(Path(state["workspace_path"]), response.content)
    
    updated_summary = get_source_overview(Path(state["workspace_path"]))
    return {"codebase_summary": updated_summary}


def integration_report_writer(state: IntegrationState):
    sys.stderr.write("\n[Phase 4 -> Writer] Generating docs/INTEGRATION_REPORT.md...\n")
    diag = state["diagnostic"]
    status = "PASSED" if (diag and diag.passed) else "FAILED"
    
    fence = "`" * 3
    report = (
        f"# Integration & End-to-End Test Report\n\n"
        f"- **Status:** {status}\n"
        f"- **Command Executed:** `{state.get('integration_test_cmd', 'pytest tests/integration -v')}`\n"
        f"- **Iterations Run:** {state.get('iteration_count', 1)}\n"
        f"- **Diagnostic Summary:** {diag.root_cause if diag else 'N/A'}\n\n"
        f"## Last Execution Output\n"
        f"{fence}text\n"
        f"{state.get('last_stdout', '')[-3000:]}\n"
        f"{fence}\n"
    )
    docs_dir = Path(state["workspace_path"]) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "INTEGRATION_REPORT.md").write_text(report, encoding="utf-8")
    return {"report_markdown": report}


# =====================================================================
# 5. CONDITIONAL ROUTING & GRAPH COMPILATION
# =====================================================================

def route_integration_result(state: IntegrationState):
    diag = state.get("diagnostic")
    iterations = state.get("iteration_count", 0)
    
    if diag and diag.passed:
        return "write_report"
    
    if iterations >= 3:
        sys.stderr.write(f"Max integration repair cycles ({iterations}) reached.\n")
        return "write_report"
        
    if diag.failure_layer == "INFRASTRUCTURE_CONFIG":
        return "fix_infra"
    elif diag.failure_layer == "TEST_SUITE":
        return "fix_tests"
    else:
        return "fix_app"


workflow = StateGraph(IntegrationState)

workflow.add_node("provisioner", environment_provisioner)
workflow.add_node("test_engineer", integration_test_engineer)
workflow.add_node("runner", integration_stack_runner)
workflow.add_node("debugger", integration_debugger)
workflow.add_node("hotfix_dev", app_hotfix_engineer)
workflow.add_node("report_writer", integration_report_writer)

workflow.add_edge(START, "provisioner")
workflow.add_edge("provisioner", "test_engineer")
workflow.add_edge("test_engineer", "runner")
workflow.add_edge("runner", "debugger")

workflow.add_conditional_edges(
    "debugger",
    route_integration_result,
    {
        "write_report": "report_writer",
        "fix_infra": "provisioner",
        "fix_tests": "test_engineer",
        "fix_app": "hotfix_dev",
    }
)
workflow.add_edge("hotfix_dev", "runner")
workflow.add_edge("report_writer", END)

integration_orchestrator = workflow.compile()


# =====================================================================
# 6. MCP TOOL REGISTRATION & SERVER ENTRY POINT
# =====================================================================

@mcp.tool()
def run_integration_and_e2e_suite(workspace_path: str = ".") -> str:
    """
    Executes Phase 4 of SDLC: Provisions a Docker Compose test environment, generates multi-service 
    E2E/API integration tests, executes them against live containers, auto-repairs infrastructure, 
    test, or source bugs, and writes docs/INTEGRATION_REPORT.md.
    
    Args:
        workspace_path: Root repository directory containing /docs and /src.
    """
    root = Path(workspace_path).resolve()
    docs_dir = root / "docs"
    
    arch_file = docs_dir / "architecture.md"
    req_file = docs_dir / "requirements.md"
    
    if not arch_file.exists():
        return "Error: docs/architecture.md not found. Run Phase 1 before Phase 4."

    initial_state: IntegrationState = {
        "workspace_path": str(root),
        "arch_text": arch_file.read_text(encoding="utf-8"),
        "requirements_text": req_file.read_text(encoding="utf-8") if req_file.exists() else "",
        "codebase_summary": get_source_overview(root),
        "env_artifacts": [],
        "test_artifacts": [],
        "integration_test_cmd": "pytest tests/integration -v",
        "last_stdout": "",
        "last_stderr": "",
        "container_logs": "",
        "last_exit_code": 0,
        "diagnostic": None,
        "report_markdown": "",
        "iteration_count": 0
    }
    
    final_state = integration_orchestrator.invoke(initial_state)
    diag = final_state.get("diagnostic")
    
    if diag and diag.passed:
        return f"Phase 4 Integration & E2E Tests PASSED! Report saved to {workspace_path}/docs/INTEGRATION_REPORT.md."
    else:
        return (
            f"Phase 4 Integration & E2E Tests FAILED after {final_state.get('iteration_count', 0)} attempts.\n"
            f"Layer: {diag.failure_layer if diag else 'UNKNOWN'} | Cause: {diag.root_cause if diag else 'N/A'}"
        )


if __name__ == "__main__":
    mcp.run()