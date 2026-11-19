import ast
import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None


# =====================================================================
# 1. SHARED SCHEMAS, EXCEPTIONS & CONCURRENCY REGISTRY
# =====================================================================

class ArtifactFile(BaseModel):
    file_path: str = Field(description="Relative file path inside workspace")
    content: str = Field(description="Full multi-line file content")


class CommandResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class TokenBudgetExceededError(RuntimeError):
    """Raised when a LangGraph thread exceeds its configured token ceiling."""
    pass


_WORKSPACE_SEMAPHORES: Dict[str, asyncio.Semaphore] = {}


def get_workspace_lock(workspace: Path) -> asyncio.Semaphore:
    """Prevents concurrent MCP calls from thrashing local GPU VRAM or colliding on files."""
    key = str(workspace.resolve())
    if key not in _WORKSPACE_SEMAPHORES:
        _WORKSPACE_SEMAPHORES[key] = asyncio.Semaphore(1)
    return _WORKSPACE_SEMAPHORES[key]


# =====================================================================
# 2. TOKEN-TRACKED LLM WRAPPER & FACTORY
# =====================================================================

class TrackedStructuredRunnable:
    def __init__(self, runnable: Any, tracker: "TrackedLLM"):
        self._runnable = runnable
        self._tracker = tracker

    def invoke(self, prompt: Any, **kwargs: Any) -> Any:
        self._tracker.check_budget()
        res = self._runnable.invoke(prompt, **kwargs)
        # Estimate tokens for structured responses when metadata isn't exposed directly
        in_est = len(str(prompt)) // 4
        out_est = len(str(res)) // 4
        self._tracker.record_usage(in_est + out_est)
        return res


class TrackedLLM:
    """Wraps any LangChain chat model to enforce per-thread token circuit breaking."""
    def __init__(self, base_llm: Any, max_tokens: int = 50000):
        self._llm = base_llm
        self.max_tokens = max_tokens
        self.total_tokens = 0

    def check_budget(self) -> None:
        if self.total_tokens >= self.max_tokens:
            raise TokenBudgetExceededError(
                f"Token circuit breaker tripped: {self.total_tokens} tokens used (limit: {self.max_tokens})."
            )

    def record_usage(self, tokens: int) -> None:
        self.total_tokens += max(0, tokens)
        self.check_budget()

    def invoke(self, prompt: Any, **kwargs: Any) -> Any:
        self.check_budget()
        res = self._llm.invoke(prompt, **kwargs)
        usage = getattr(res, "usage_metadata", None) or {}
        used = usage.get("total_tokens")
        if used is None:
            content = getattr(res, "content", str(res))
            used = (len(str(prompt)) + len(str(content))) // 4
        self.record_usage(int(used))
        return res

    def with_structured_output(self, schema: Any, **kwargs: Any) -> TrackedStructuredRunnable:
        return TrackedStructuredRunnable(self._llm.with_structured_output(schema, **kwargs), self)


def get_llm(model_name: Optional[str] = None, num_ctx: int = 4096, max_tokens: Optional[int] = None) -> TrackedLLM:
    """Centralized local Ollama model factory wrapped with token budget tracking."""
    selected = model_name or os.getenv("SDLC_OLLAMA_MODEL", "qwen2.5-coder:7b")
    budget = max_tokens or int(os.getenv("SDLC_MAX_TOKENS_PER_RUN", "50000"))
    base = ChatOllama(
        model=selected,
        temperature=0.0,
        num_ctx=num_ctx,
        keep_alive="15m",
        client_kwargs={"timeout": 120.0},
    )
    return TrackedLLM(base, max_tokens=budget)


# =====================================================================
# 3. PATH-SAFE MARKDOWN FILE EXTRACTION & AST GATES
# =====================================================================

def extract_and_write_files(workspace: Path, llm_text: str) -> List[ArtifactFile]:
    """
    Parses ```lang blocks preceded by 'FILE: path/to/file' and writes clean multi-line files.
    Enforces strict path-traversal containment within `workspace`.
    """
    pattern = r"FILE:\s*([^\n`]+)\s*\n```[a-zA-Z0-9_-]*\n(.*?)```"
    matches = re.findall(pattern, llm_text, flags=re.DOTALL)
    written: List[ArtifactFile] = []
    root = workspace.resolve()

    for rel_path, code in matches:
        clean_path = rel_path.strip().strip("`*")
        target = (root / clean_path).resolve()

        if not target.is_relative_to(root):
            raise ValueError(f"Security violation: blocked path traversal attempt -> '{clean_path}'")

        target.parent.mkdir(parents=True, exist_ok=True)
        clean_code = code.strip() + "\n"
        target.write_text(clean_code, encoding="utf-8")
        sys.stderr.write(f"[Core] Wrote file: {clean_path}\n")
        written.append(ArtifactFile(file_path=clean_path, content=clean_code))

    if not written:
        sys.stderr.write("[Core] Warning: No 'FILE: <path>' blocks matched in LLM response.\n")
    return written


def validate_python_ast(files: List[ArtifactFile]) -> Optional[str]:
    """Runs zero-token CPU AST syntax validation on all Python artifacts before test execution."""
    for f in files:
        if f.file_path.endswith(".py"):
            try:
                ast.parse(f.content, filename=f.file_path)
            except SyntaxError as e:
                return f"SyntaxError in {f.file_path} at line {e.lineno}: {e.msg}"
    return None


# =====================================================================
# 4. CODEBASE & EXISTING TEST CONTEXT LOADERS
# =====================================================================

def get_source_snapshot(workspace: Path, max_chars: int = 10000) -> str:
    """Collects non-test source files under src/ (or project root) for agent context."""
    ignore_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", "docs", ".pytest_cache", ".worktrees", ".orchestrator"}
    allowed_exts = {".py", ".ts", ".js", ".go", ".sql", ".yml", ".yaml"}
    chunks = []
    scan_root = (workspace / "src") if (workspace / "src").exists() else workspace

    for path in sorted(scan_root.rglob("*")):
        if any(part in ignore_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix in allowed_exts and "test" not in path.parts:
            rel = path.relative_to(workspace).as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks.append(f"### FILE: {rel}\n{text[:3000]}")
    return "\n\n".join(chunks)[:max_chars]


def get_existing_tests(workspace: Path, exclude_dirs: Tuple[str, ...] = ("regression",), max_chars: int = 3500) -> str:
    """Collects existing unit/integration tests so hotfixes and stories preserve backward compatibility."""
    tests_dir = workspace / "tests"
    if not tests_dir.exists():
        return ""
    chunks = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        if any(ex in path.parts for ex in exclude_dirs):
            continue
        rel = path.relative_to(workspace).as_posix()
        chunks.append(f"### {rel}\n{path.read_text(encoding='utf-8', errors='ignore')[:1500]}")
    return "\n\n".join(chunks)[:max_chars]


# =====================================================================
# 5. SCRUBBED-ENV SUBPROCESS & PYTEST RUNNER
# =====================================================================

def build_scrubbed_env(workspace: Path, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Constructs a minimal environment containing only safe system variables and dual-path PYTHONPATH,
    preventing untrusted test runs from reading host API keys or cloud tokens.
    """
    safe_keys = ("PATH", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP", "HOME", "USERPROFILE", "VIRTUAL_ENV")
    clean_env = {k: v for k, v in os.environ.items() if k in safe_keys}
    clean_env["PYTHONPATH"] = f"{workspace.resolve()}{os.pathsep}{(workspace / 'src').resolve()}"
    clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_env:
        clean_env.update(extra_env)
    return clean_env


def run_sandboxed_tests(
    workspace: Path,
    command: Optional[str] = None,
    timeout_seconds: int = 30,
    extra_env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    """Executes pytest (or custom test runner) with scrubbed env, dual PYTHONPATH, and fast-fail."""
    env = build_scrubbed_env(workspace, extra_env=extra_env)

    if command is None:
        cmd: object = [sys.executable, "-m", "pytest", "-v", "--tb=short", "--maxfail=1"]
        use_shell = False
    else:
        cmd_str = command
        if "pytest" in cmd_str and "--maxfail" not in cmd_str:
            cmd_str = f"{cmd_str} --maxfail=1"
        cmd = cmd_str
        use_shell = True

    try:
        proc = subprocess.run(
            cmd,
            shell=use_shell,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return CommandResult(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return CommandResult(
            stdout=out,
            stderr=f"Test execution timed out after {timeout_seconds}s (check for lock deadlocks or infinite loops).",
            exit_code=1,
            timed_out=True,
        )
    except Exception as e:
        return CommandResult(stdout="", stderr=f"Execution exception: {str(e)}", exit_code=1)


# =====================================================================
# 6. GIT WORKTREE ISOLATION
# =====================================================================

@contextmanager
def git_worktree_sandbox(repo_root: Path, branch_slug: str):
    """
    Creates an isolated git worktree under `.worktrees/<branch_slug>` if `repo_root` is a Git repo.
    Falls back to `repo_root` directly if Git is not initialized yet.
    """
    root = repo_root.resolve()
    if not (root / ".git").exists():
        yield root
        return

    safe_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", branch_slug).strip("-")
    worktree_dir = root / ".worktrees" / safe_slug
    branch_name = f"sdlc/{safe_slug}"

    if worktree_dir.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree_dir)], cwd=root, capture_output=True)

    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    add_proc = subprocess.run(
        ["git", "worktree", "add", "-B", branch_name, str(worktree_dir)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if add_proc.returncode != 0:
        sys.stderr.write(f"[Core] Worktree fallback to main workspace: {add_proc.stderr.strip()}\n")
        yield root
        return

    sys.stderr.write(f"[Core] Isolated worktree active: {worktree_dir} (branch: {branch_name})\n")
    try:
        yield worktree_dir
    finally:
        sys.stderr.write(f"[Core] Keeping worktree at {worktree_dir} for review/PR merge.\n")


# =====================================================================
# 7. LANGGRAPH SQLITE CHECKPOINTER & JSONL AUDIT LOGGING
# =====================================================================

def get_sqlite_checkpointer(workspace: Path):
    """
    Returns a hardened SqliteSaver backed by `.orchestrator/checkpoints.sqlite`.
    Enforces strict msgpack deserialization for security.
    """
    if SqliteSaver is None:
        return None
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    state_dir = workspace.resolve() / ".orchestrator"
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "checkpoints.sqlite"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def log_audit_event(
    workspace: Path,
    phase: str,
    thread_id: str,
    status: str,
    iterations: int,
    details: Optional[Dict[str, object]] = None,
) -> None:
    """Appends structured JSONL execution telemetry to `.orchestrator/audit.jsonl`."""
    audit_dir = workspace.resolve() / ".orchestrator"
    audit_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "thread_id": thread_id,
        "status": status,
        "iterations": iterations,
        "details": details or {},
    }
    with (audit_dir / "audit.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# =====================================================================
# 8. STAGE 2: DYNAMIC MARKDOWN SPEC & SKILL SLICER (O(1) TOKEN LOADER)
# =====================================================================

_SPEC_CACHE: Dict[str, Tuple[float, Dict[str, str]]] = {}


def _parse_markdown_sections(md_path: Path) -> Dict[str, str]:
    """Parses top-level `## <section_key>` blocks from a Markdown file with mtime caching."""
    if not md_path.exists():
        return {}
    resolved = str(md_path.resolve())
    mtime = md_path.stat().st_mtime
    cached = _SPEC_CACHE.get(resolved)
    if cached and cached[0] == mtime:
        return cached[1]

    text = md_path.read_text(encoding="utf-8")
    sections: Dict[str, str] = {}
    current_key: Optional[str] = None
    buffer: List[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_key is not None:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = line[3:].strip()
            buffer = []
        elif current_key is not None:
            buffer.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(buffer).strip()

    _SPEC_CACHE[resolved] = (mtime, sections)
    return sections


def _resolve_spec_file(filename: str, workspace: Optional[Path] = None) -> Path:
    """Checks target workspace first for project-specific overrides, then falls back to orchestrator root."""
    if workspace is not None:
        override = workspace.resolve() / filename
        if override.exists():
            return override
    return Path(__file__).resolve().parent / filename


def load_agent_prompt(
    agent_id: str,
    skill_ids: Optional[List[str]] = None,
    workspace: Optional[Path] = None,
    include_retry_skill: bool = False,
) -> str:
    """
    Loads ONLY the requested agent persona from `agents.md` and requested skill slices from `skills.md`.
    Automatically appends `retry_recovery` only when `include_retry_skill=True` (saving tokens on Pass #1).
    """
    agents_map = _parse_markdown_sections(_resolve_spec_file("agents.md", workspace))
    skills_map = _parse_markdown_sections(_resolve_spec_file("skills.md", workspace))

    agent_block = agents_map.get(agent_id, f"You are the {agent_id} agent.")
    parts = [f"[Role & Rules: {agent_id}]\n{agent_block}"]

    requested_skills = list(skill_ids or [])
    if include_retry_skill and "retry_recovery" not in requested_skills:
        requested_skills.append("retry_recovery")

    for sid in requested_skills:
        skill_text = skills_map.get(sid)
        if skill_text:
            parts.append(f"[Skill Contract: {sid}]\n{skill_text}")

    return "\n\n".join(parts)