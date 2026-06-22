# ObjGauss Claude Code Instructions

Use the same workflow as every other AI agent in this repository.

Read first:

- `docs/development-flow.md`

Do not maintain a separate Claude-specific process. If the workflow needs to
change, update `docs/development-flow.md` and keep this file as a pointer.

Default validation:

```bash
uv run --extra dev pytest
npm run build
```
