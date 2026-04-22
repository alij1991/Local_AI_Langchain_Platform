# Systems module — landmines & architecture

"Systems" = user-defined multi-agent workflows (DAGs) + hardcoded pre-built templates.

## Where things live

- `system_templates.py` — hardcoded, immutable pre-built templates (code-only, no DB).
- `system_info.py` — execution engine: validates DAG, runs agents in topological order.
- `repositories/systems.py` — CRUD for user-defined systems (SQLite `data/app.db`).
- `flutter_client/lib/pages/systems_page.dart` — UI.

## Landmines

1. **Templates are hardcoded.** Adding/editing a pre-built template = code change + restart. There is no admin UI. Do NOT suggest "edit it in settings".
2. **Orphaned edges are silently dropped on load.** If a node is deleted but an edge referencing it remains, `load_system()` quietly removes the edge without logging. Silent data loss — if a user complains "my connection disappeared", this is why.
3. **No schema validation on `definition`.** The `definition` JSON blob is stored as-is. Garbage in → runtime errors deep in the executor. When accepting user input, validate structure up front.
4. **Timestamps are UTC ISO strings, not Unix epoch.** Don't compare with `time.time()`. Use `datetime.fromisoformat()`.
5. **Agent names in edges are free-text, not FK.** If a template references an agent that doesn't exist at runtime, the node fails silently and downstream nodes receive `None`. Always check agent exists before executing.
6. **Cycle detection uses Kahn's algorithm.** A cycle returns empty topological order → system runs zero nodes and "succeeds". If a run mysteriously produces no output, check for cycles first.

## Conventions

- Log tag: `[SYSTEM]` (not `[SYSTEMS]`).
- DAG node IDs are UUIDs generated client-side in Flutter — do not regenerate server-side.
- Execution is sequential (not parallel) even when DAG allows it. This is intentional — keeps token budget predictable.

## Don't-fix

- The silent orphaned-edge cleanup is load-bearing for template migrations. Adding logging is fine; removing the cleanup is not.
- Sequential execution is intentional. Don't "optimize" to parallel without discussion.
