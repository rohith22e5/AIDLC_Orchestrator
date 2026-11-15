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
# 1. INITIALIZE MCP & MODEL
# =====================================================================

mcp = FastMCP("SDLC-Implementation-Orchestrator")

# Fast, high-reasoning model for coding loops
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.0)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0)


# =====================================================================
# 2. SCHEMAS & STATE
# =====================================================================

class FileModification(BaseModel):
    file_path: str = Field(description="Relative path to file, e.g., src/services/auth.py or tests/test_auth.py")
    content: str = Field(description="Full complete code content of the file")

class DebugFeedback(BaseModel):
    test_passed: bool
    target_layer: str = Field(
        description="'NONE', 'TEST_FILE' (if tests have bad mocks, wrong imports, or invalid assertions), or 'SOURCE_FILE' (if src/ has a bug)"
    )
    diagnostic_analysis: str = Field(description="Root cause of runtime or assertion failure")
    suggested_fix: str

class ImplementationState(TypedDict):
    workspace_path: str
    target_story_id: str
    test_runner_command: str
    plan_text: str
    arch_text: str
    requirements_text: str
    
    test_files: List[FileModification]
    source_files: List[FileModification]
    
    last_run_stdout: str
    last_run_stderr: str
    last_run_exit_code: int
    
    debug_feedback: Optional[DebugFeedback]
    iteration_count: int


# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================

def extract_and_write_files(workspace_root: Path, llm_text: str) -> List[FileModification]:
    """Parses ```lang blocks preceded by 'FILE: path/to/file' and writes clean multi-line files."""
    pattern = r"FILE:\s*([^\n`]+)\s*\n```[a-zA-Z0-9_-]*\n(.*?)```"
    matches = re.findall(pattern, llm_text, flags=re.DOTALL)
    written: List[FileModification] = []

    for rel_path, code in matches:
        clean_path = rel_path.strip().strip("`*")
        target = workspace_root / clean_path
        target.parent.mkdir(parents=True, exist_ok=True)
        clean_code = code.strip() + "\n"
        target.write_text(clean_code, encoding="utf-8")
        sys.stderr.write(f"Updated: {clean_path}\n")
        written.append(FileModification(file_path=clean_path, content=clean_code))

    if not written:
        sys.stderr.write("Warning: No 'FILE: <path>' code blocks matched in LLM response.\n")
    return written


# =====================================================================
# 4. AGENT NODES
# =====================================================================

def test_engineer(state: ImplementationState):
    sys.stderr.write("\n[Agent: Test Engineer] Authoring/Updating Test Suite...\n")
    
    debug_context = ""
    fb = state.get("debug_feedback")
    if fb and not fb.test_passed and fb.target_layer == "TEST_FILE":
        existing_tests = "\n\n".join(f"# {f.file_path}\n{f.content}" for f in state.get("test_files", []))
        debug_context = (
            f"\nPREVIOUS TEST SUITE BUG TO FIX:\n"
            f"Diagnostic: {fb.diagnostic_analysis}\n"
            f"Fix Instructions: {fb.suggested_fix}\n"
            f"Error Output:\n{(state['last_run_stderr'] or state['last_run_stdout'])[-1500:]}\n"
            f"Existing Test Code:\n{existing_tests[:2000]}\n"
            f"Test public behavioral contracts—do NOT mock internal methods unless explicitly required.\n"
        )

    fence = "`" * 3
    prompt = f"""
    You are a Senior Test Engineer practicing Test-Driven Development (TDD).
    
    Target User Story ID: {state['target_story_id']}
    
    Context:
    Architecture:
    {state['arch_text'][:3000]}
    
    Execution Plan / User Stories:
    {state['plan_text'][:3000]}
    {debug_context}

    Task:
    Generate comprehensive automated test files (e.g. pytest or jest) that assert 
    the acceptance criteria of the target story and its underlying API/data contracts.
    Include happy path and edge-case assertions, and keep any sleep/timer tests fast (< 0.2s).

    Output EVERY file strictly using this Markdown format (do NOT output JSON):

    FILE: tests/test_solution.py
    {fence}python
    # complete multi-line test code here
    {fence}
    """
    
    response = llm.invoke(prompt)
    workspace_root = Path(state["workspace_path"])
    files = extract_and_write_files(workspace_root, response.content)
    
    return {"test_files": files}


def software_engineer(state: ImplementationState):
    sys.stderr.write("\n[Agent: Software Engineer] Writing Source Implementation...\n")
    
    test_contents = "\n\n".join(f"# {f.file_path}\n{f.content}" for f in state.get("test_files", []))
    debug_context = ""
    fb = state.get("debug_feedback")
    if fb and not fb.test_passed and fb.target_layer == "SOURCE_FILE":
        debug_context = (
            f"\nPREVIOUS EXECUTION FAILURE:\n"
            f"Exit Code: {state['last_run_exit_code']}\n"
            f"Error Output:\n"
            f"{(state['last_run_stderr'] or state['last_run_stdout'])[-2000:]}\n"
            f"Diagnostic: {fb.diagnostic_analysis}\n"
            f"Remediation: {fb.suggested_fix}\n"
        )

    fence = "`" * 3
    prompt = f"""
    You are a Principal Software Development Engineer.
    Target Story: {state['target_story_id']}
    
    Context:
    Architecture:
    {state['arch_text'][:3500]}
    
    Generated Test Code:
    {test_contents[:3000]}
    {debug_context}

    Task:
    Generate or fix the exact production application source code so that all tests pass.
    Ensure code is clean, thread-safe (use reentrant locks like `threading.RLock` if nesting calls), 
    and follows the specified system architecture.

    Output EVERY file strictly using this Markdown format (do NOT output JSON):

    FILE: src/solution.py
    {fence}python
    # complete multi-line source code here
    {fence}
    """
    
    response = llm.invoke(prompt)
    workspace_root = Path(state["workspace_path"])
    files = extract_and_write_files(workspace_root, response.content)
    
    return {"source_files": files}


def test_runner_node(state: ImplementationState):
    sys.stderr.write(f"\n[Environment: Test Runner] Running command: {state['test_runner_command']}...\n")
    workspace = Path(state["workspace_path"])

    if not state.get("source_files"):
        err_msg = "No source files were extracted from the Software Engineer's output."
        sys.stderr.write(f"{err_msg}\n")
        return {"last_run_stdout": "", "last_run_stderr": err_msg, "last_run_exit_code": 1}

    # Fast AST syntax validation before running test subprocess
    for f in state.get("source_files", []) + state.get("test_files", []):
        if f.file_path.endswith(".py"):
            try:
                ast.parse(f.content)
            except SyntaxError as e:
                err_msg = f"SyntaxError in {f.file_path} at line {e.lineno}: {e.msg}"
                sys.stderr.write(f"{err_msg}\n")
                return {"last_run_stdout": "", "last_run_stderr": err_msg, "last_run_exit_code": 1}

    env = os.environ.copy()
    env["PYTHONPATH"] = str((workspace / "src").resolve())
    cmd = state["test_runner_command"]
    if "pytest" in cmd and "--maxfail" not in cmd:
        cmd = f"{cmd} --maxfail=1"
    
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=45
        )
        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode("utf-8", errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = "Test runner timed out after 45s (check for lock deadlocks or infinite loops)."
        exit_code = 1
    except Exception as e:
        stdout = ""
        stderr = f"Command execution exception: {str(e)}"
        exit_code = 1

    sys.stderr.write(f"Execution finished with exit code {exit_code}\n")
    return {
        "last_run_stdout": stdout,
        "last_run_stderr": stderr,
        "last_run_exit_code": exit_code
    }


def debugger_agent(state: ImplementationState):
    sys.stderr.write("\n[Agent: Debugger / Quality Gate] Analyzing test suite results...\n")
    
    exit_code = state["last_run_exit_code"]
    stdout = state["last_run_stdout"]
    stderr = state["last_run_stderr"]
    
    if exit_code == 0:
        sys.stderr.write("Verification SUCCESS: All tests passed.\n")
        feedback = DebugFeedback(
            test_passed=True,
            target_layer="NONE",
            diagnostic_analysis="All unit tests and assertions passed successfully.",
            suggested_fix="None"
        )
    else:
        sys.stderr.write("Verification FAILED. Diagnosing failure logs...\n")
        test_code = "\n\n".join(f"# {f.file_path}\n{f.content}" for f in state.get("test_files", []))
        src_code = "\n\n".join(f"# {f.file_path}\n{f.content}" for f in state.get("source_files", []))

        prompt = f"""
        You are an elite Staff Debugger. The tests failed.
        
        Command Run: {state['test_runner_command']}
        Exit Code: {exit_code}
        Stdout:
        {stdout[-2000:]}
        Stderr:
        {stderr[-2000:]}

        [Test Suite]
        {test_code[:2000]}

        [Application Source]
        {src_code[:2000]}
        
        Determine whether the bug is in the test suite (`TEST_FILE` - e.g., bad mocks, wrong import names, 
        or testing unrequested internals) or in the application source (`SOURCE_FILE` - e.g., logic bugs, 
        lock deadlocks, syntax errors, unhandled exceptions).
        Set `target_layer` strictly to 'TEST_FILE' or 'SOURCE_FILE' and provide an exact diagnosis and fix.
        """
        feedback = llm.with_structured_output(DebugFeedback).invoke(prompt)

    new_iteration = state.get("iteration_count", 0) + 1
    return {
        "debug_feedback": feedback,
        "iteration_count": new_iteration
    }


# =====================================================================
# 5. CONDITIONAL ROUTING & GRAPH COMPILATION
# =====================================================================

def evaluate_build_health(state: ImplementationState):
    feedback = state.get("debug_feedback")
    iterations = state.get("iteration_count", 0)
    
    if feedback and feedback.test_passed:
        return "success"
    
    if iterations >= 3:
        sys.stderr.write(f"Reached maximum repair cycles ({iterations}). Halting.\n")
        return "max_iterations"

    if feedback and feedback.target_layer == "TEST_FILE":
        return "fix_test"
        
    return "fix_source"


workflow = StateGraph(ImplementationState)

workflow.add_node("test_engineer", test_engineer)
workflow.add_node("software_engineer", software_engineer)
workflow.add_node("test_runner", test_runner_node)
workflow.add_node("debugger", debugger_agent)

workflow.add_edge(START, "test_engineer")
workflow.add_edge("test_engineer", "software_engineer")
workflow.add_edge("software_engineer", "test_runner")
workflow.add_edge("test_runner", "debugger")

workflow.add_conditional_edges(
    "debugger",
    evaluate_build_health,
    {
        "success": END,
        "fix_test": "test_engineer",
        "fix_source": "software_engineer",
        "max_iterations": END
    }
)

implementation_orchestrator = workflow.compile()


# =====================================================================
# 6. FASTMCP TOOL EXPOSURE
# =====================================================================

@mcp.tool()
def implement_and_verify_story(
    target_story_id: str,
    test_runner_command: str = "pytest",
    workspace_path: str = "."
) -> str:
    """
    Executes Phase 2 of SDLC: Ingests the spec documents from /docs, writes automated tests, 
    implements the source code, runs the test runner, and iteratively fixes errors until tests pass.
    
    Args:
        target_story_id: The User Story ID to implement from docs/plan.md (e.g., 'US-01' or 'ALL').
        test_runner_command: The terminal command to verify code (e.g. 'pytest', 'npm test', 'go test ./...').
        workspace_path: Root repository directory.
    """
    root = Path(workspace_path)
    docs_dir = root / "docs"
    
    plan_file = docs_dir / "plan.md"
    arch_file = docs_dir / "architecture.md"
    req_file = docs_dir / "requirements.md"
    
    if not (plan_file.exists() and arch_file.exists()):
        return (
            "Error: Phase 1 documents (docs/plan.md and docs/architecture.md) not found. "
            "Please run Phase 1 (generate_project_specifications) first."
        )

    initial_state: ImplementationState = {
        "workspace_path": str(root.resolve()),
        "target_story_id": target_story_id,
        "test_runner_command": test_runner_command,
        "plan_text": plan_file.read_text(encoding="utf-8"),
        "arch_text": arch_file.read_text(encoding="utf-8"),
        "requirements_text": req_file.read_text(encoding="utf-8") if req_file.exists() else "",
        "test_files": [],
        "source_files": [],
        "last_run_stdout": "",
        "last_run_stderr": "",
        "last_run_exit_code": 0,
        "debug_feedback": None,
        "iteration_count": 0
    }
    
    final_output = implementation_orchestrator.invoke(initial_state)
    feedback = final_output.get("debug_feedback")
    
    if feedback and feedback.test_passed:
        return f"Successfully implemented {target_story_id}! All tests passed with command '{test_runner_command}'."
    else:
        err = final_output.get('last_run_stderr') or final_output.get('last_run_stdout')
        return (
            f"Implemented {target_story_id}, but tests are failing after retry limit.\n"
            f"Layer: {feedback.target_layer if feedback else 'UNKNOWN'} | "
            f"Diagnostic: {feedback.diagnostic_analysis if feedback else 'Unknown'}\n"
            f"Last Error Trace: {err[-1000:] if err else 'None'}"
        )


if __name__ == "__main__":
    mcp.run()