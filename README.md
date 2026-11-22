# AIDLC_Orchestrator

An AI-Driven Software Development Lifecycle (AIDLC) orchestration engine built with **Python**, **LangGraph**, **FastMCP**, **`pytest`**, **Git worktrees**, and **local Ollama models**—designed to automate spec generation, test-driven code implementation, review/integration, and human-in-the-loop checkpoints with minimal or zero cloud API spend.

## Overview

`AIDLC_Orchestrator` coordinates specialized sub-graphs and MCP tool servers across an end-to-end engineering workflow while keeping execution isolated in Git worktrees and grounded in repository specifications (`AGENTS.md`, `SKILLS.md`, `CLAUDE.md`, and `.aidlc/` artifacts).

### Key Capabilities

- **Spec-Driven Phase 1 (Requirements & Architecture):** Generates structured specifications and design artifacts before implementation begins, guided by project rules and agent instructions.
- **TDD Implementation & Automated Repair Loops (Phase 2):** Writes code and `pytest` suites, runs test verification, and loops through automated repair steps on test or import failures.
- **Multi-Stage Sub-Graph Orchestration:** Coordinates requirements, coding, review, integration, and DevOps stages from `master-orchestrator.py` with explicit state transitions and human-in-the-loop decision checkpoints.
- **Git Worktree Isolation:** Runs parallel or sandboxed feature/task worktrees so experimental edits and automated repair loops do not clobber your main working branch.
- **Local-First Model Execution:** Routes tasks to local models via **Ollama** (such as `qwen2.5-coder:7b`, `deepseek-coder-v2`, `gemma4:e4b`, or `granite4.2:8b`) with optional hybrid fallback to external coding agents/models.
- **FastMCP Tool Integration:** Exposes deterministic filesystem, Git, test-runner, and spec tools over MCP so LangGraph nodes interact with the workspace cleanly.

## Architecture

```text
                       ┌─────────────────────────┐
                       │  master-orchestrator.py │
                       │   (LangGraph State &    │
                       │   HITL Checkpoints)     │
                       └────────────┬────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│ Spec & Design │           │ TDD Coding &  │           │ Review, Integ │
│   Sub-Graph   │ ────────► │  Repair Loop  │ ────────► │ & DevOps Graph│
└───────┬───────┘           └───────┬───────┘           └───────┬───────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ FastMCP Servers & Git Worktree│
                    │  (pytest, file/git ops, etc.) │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Ollama Local Runtime      │
                    │  (Code, Reasoning, Embeddings)│
                    └───────────────────────────────┘
```

## Project Structure

```text
AIDLC_Orchestrator/
├── master-orchestrator.py   # Top-level LangGraph workflow & HITL checkpoints
├── AGENTS.md                # Repository agent guidelines & conventions
├── SKILLS.md                # Reusable agent capabilities & tool usage rules
├── .aidlc/                  # Generated requirements, specs, and workflow state
├── mcp_servers/             # FastMCP server implementations
├── graphs/                  # LangGraph sub-graphs (spec, coding, review, integration, devops)
├── tests/                   # Orchestrator & generated service test suites (pytest)
├── pyproject.toml           # Dependencies & project configuration
└── README.md
```

## Prerequisites

- **Python** 3.11+ (managed via `uv` or `pyenv`)
- **Git** (with support for `git worktree`)
- **Ollama** installed and running locally (`ollama serve`)
- One or more local models pulled in Ollama, for example:
  ```bash
  ollama pull qwen2.5-coder:7b
  ollama pull gemma4:e4b
  ollama pull nomic-embed-text
  ```

## Installation

Clone the repository and install dependencies using `uv` (or `pip`):

```bash
git clone [https://github.com/rohith22e5/AIDLC_Orchestrator.git](https://github.com/rohith22e5/AIDLC_Orchestrator.git)
cd AIDLC_Orchestrator

# Using uv (recommended)
uv venv
uv sync

# Or using standard pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

### 1. Start Ollama

Ensure the local Ollama daemon is running and serving models:

```bash
ollama serve
```

### 2. Run the Master Orchestrator

Execute `master-orchestrator.py` against a target feature prompt or workspace:

```bash
uv run python master-orchestrator.py --task "Implement a rate limiter service with unit tests"
```

*(Update CLI flags above if `master-orchestrator.py` uses positional arguments, config files, or environment variables.)*

### 3. Human-in-the-Loop (HITL) Checkpoints

When the graph reaches a checkpoint (such as approving generated specs, reviewing diffs, or resolving repeated test failures), inspect the generated artifacts in `.aidlc/` and the active Git worktree before continuing or adjusting the state.

## Running Tests

Run the test suite with `pytest`:

```bash
uv run pytest
```

If you encounter import resolution issues when running `pytest` on generated or nested modules, run with `python -m pytest` or verify package/path configuration in `pyproject.toml` and `conftest.py`.

## Configuration & Spec Conventions

- **`AGENTS.md` / `CLAUDE.md`:** Define coding standards, layer boundaries, and agent rules that sub-graphs load during planning and implementation.
- **`SKILLS.md`:** Document domain workflows, testing expectations, and MCP tool patterns.
- **`.aidlc/`:** Stores phase outputs (requirements, design specs, execution plans) so downstream graphs and human reviewers have a single source of truth.
- **Git Worktrees:** Isolate agent runs per task/branch so you can review or discard changes cleanly without polluting your primary checkout.

## Roadmap / Future Enhancements

- Tighten token/context budgets across sub-graphs with structured state handoffs and concise spec summaries.
- Expand automated `pytest` repair loops for multi-file imports, fixtures, and edge-case regressions.
- Add hybrid routing rules to fall back selectively to stronger external models only on complex architectural or debugging steps.
- Harden Git worktree cleanup, merge/rebase hooks, and CI checks in the integration/DevOps sub-graph.
