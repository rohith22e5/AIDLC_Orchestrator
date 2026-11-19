# Autonomous SDLC Orchestrator Workspace Rules

When working in this repository or any target workspace managed by these MCP servers:
1. Follow the agent persona definitions in `agents.md` and technical contracts in `skills.md`.
2. Always include both `.` and `src/` in `PYTHONPATH` when running `pytest -v --tb=short --maxfail=1`.
3. Validate syntax with `python -m py_compile <file>` before executing test suites.
4. Preserve exact exception types (`TypeError` for invalid types/bools, `ValueError` for out-of-range values) and existing error message strings asserted by `tests/`.
5. Never write files outside the active workspace or Git worktree root.