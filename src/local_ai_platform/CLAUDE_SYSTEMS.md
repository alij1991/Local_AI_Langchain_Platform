# Systems module — landmines & architecture

"Systems" = user-defined multi-agent workflows (DAGs) + hardcoded pre-built templates.

## Where things live

- `system_templates.py` — hardcoded, immutable pre-built templates (code-only, no DB).
- `agents.py::AgentOrchestrator.execute_system_graph` — **actual** DAG executor (~L990). BFS walker with a `visited` set + `max_steps = 2 * len(nodes)` safety cap. Not a topological sort.
- `repositories/systems.py` — CRUD for user-defined systems (SQLite `data/app.db`). Plain SELECT/upsert — no validation or cleanup of the stored `definition_json`.
- `flutter_client/lib/pages/systems_page.dart` — UI (designer + run + templates tabs).
- `system_info.py` — hardware detection + model recommendations (CPU/GPU/VRAM). **Unrelated to the systems feature** despite the shared prefix; do not look here for the DAG engine.

## Landmines

1. **Templates are hardcoded.** Adding/editing a pre-built template = code change + restart. There is no admin UI. Do NOT suggest "edit it in settings".
2. **Orphaned edges are silently dropped on load — client-side only.** The filter lives in `flutter_client/lib/pages/systems_page.dart` (~L185-187, `ed.removeWhere(...)`), not in the server. `repositories/systems.py::get_system` returns the stored JSON verbatim, so raw API consumers (or any non-Flutter client) will see orphan edges. Silent data loss from the user's point of view — if a user complains "my connection disappeared", this is why. See [IMPROVE-31] for the proposed server-side fix.
3. **No schema validation on `definition`.** The `definition` JSON blob is stored as-is. Garbage in → runtime errors deep in the executor. When accepting user input, validate structure up front.
4. **Timestamps are UTC ISO strings, not Unix epoch.** Don't compare with `time.time()`. Use `datetime.fromisoformat()`.
5. **Agent names in edges are free-text, not FK.** If a template references an agent that doesn't exist at runtime, the node fails silently and downstream nodes receive `None`. Always check agent exists before executing.
6. **No explicit cycle detection.** The BFS walker in `execute_system_graph` guards against re-entry with a `visited` set; `max_steps = 2 * len(nodes)` is a belt-and-suspenders cap that almost never trips. A cycle like `A→B→A` executes A once, then B once, then stops — and the run reports `status="ok"` with whichever nodes it reached. If a cyclic graph "succeeds" but produces only partial output, this is why. Kahn-based pre-flight cycle rejection on save is planned under [IMPROVE-37] but not shipped.

## Conventions

- Log tag: `[SYSTEM]` (not `[SYSTEMS]`).
- DAG node IDs are UUIDs generated client-side in Flutter — do not regenerate server-side.
- Execution is sequential (not parallel) even when DAG allows it. This is intentional — keeps token budget predictable.

## Don't-fix

- The silent orphaned-edge cleanup is load-bearing for template migrations. Adding logging is fine; removing the cleanup is not.
- Sequential execution is intentional. Don't "optimize" to parallel without discussion.
