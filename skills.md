# SDLC Reusable Skills & Contracts

## markdown_file_output
Output EVERY created or modified file strictly using this exact Markdown fence format (do NOT output raw JSON):

FILE: relative/path/to/file.ext
```language
# complete multi-line file content here
```

## python_validation_contracts
- Python imports: Modules inside `src/` are on `PYTHONPATH`; import directly (e.g., `from cache import TTLCache`), never `from src.cache import ...` unless `src/__init__.py` exists.
- Numeric vs. Boolean guard: In Python `isinstance(True, int)` is `True`. Always check `if isinstance(val, bool) or not isinstance(val, (int, float)): raise TypeError(...)` before range checks (`if val <= 0: raise ValueError(...)`).
- Concurrency: Always use `threading.RLock()` instead of `threading.Lock()` to prevent reentrant deadlocks when helper methods acquire the same lock.
- Deterministic tests: Keep all `time.sleep()` durations under `0.2s` so test suites complete within seconds.

## docker_devops_contracts
- Multi-stage builds: Use `python:3.13-slim` (or matching project Python version), install dependencies in a builder stage, and run the final container as non-root `appuser`.
- Compose isolation: Never bind hardcoded host ports in `docker-compose.test.yml` that could collide with running services; always include `healthcheck` blocks and ephemeral volumes.
- CI hygiene: Pin major action versions in `.github/workflows/ci.yml` (`actions/checkout@v4`, `astral-sh/setup-uv@v4`) and run `pytest -v --tb=short`.

## sre_incident_contracts
- Regression isolation: Write incident reproduction tests under `tests/regression/test_incident_regression.py` so they run alongside unit and integration suites without overwriting existing tests.
- Backward compatibility: Never break existing public signatures or alter exception messages asserted by `tests/`.
- Blameless post-mortems: Include Executive Summary, 5-Whys Root Cause Analysis, Verification Proof, and Action Items with zero bracketed placeholders.

## retry_recovery
- Analyze the exact pytest, AST, or config-validator traceback carefully before editing code.
- If an `AssertionError` shows a mismatch between your raised exception message and an existing test's `assert str(e) == "..."`, update your source code to emit the exact message expected by the existing test suite.
- Ensure BOTH existing unit/integration tests AND newly added regression/story tests pass simultaneously.