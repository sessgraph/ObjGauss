# ObjGauss AI Agent Instructions

This repository uses one shared development workflow for Codex, Claude Code,
and other AI coding sessions.

Read this first:

- `docs/development-flow.md`

Before changing files:

1. Check `git status --short`.
2. Read `docs/state/project-status.md` and `docs/state/pr-queue.md` if they exist.
3. State the target files, goal, scope, and validation plan.
4. Keep training assets, demo assets, and generated outputs separated.

Default validation:

```bash
uv run --extra dev pytest
npm run build
```
