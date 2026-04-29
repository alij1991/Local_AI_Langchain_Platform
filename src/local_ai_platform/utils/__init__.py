"""[IMPROVE-111] Cross-cutting helpers package.

The ``utils`` namespace holds helpers that don't fit into a
specific subsystem package (images, partner, providers, etc.)
because they're consumed across multiple subsystems. Today the
package contains:

  * ``validation`` — signature-based kwarg validation helpers.
    Promoted from partner/export.py:_validate_decay_config_keys
    so other JSON-payload callsites (memory config, image
    options, future bundle metadata) can reuse the same
    inspect.signature-driven validator.

Why a NEW package rather than dropping helpers into existing
modules: each new helper here is a candidate for cross-subsystem
reuse. Putting them in observability_events.py, partner/export.py,
or images/service.py would couple the helper to that subsystem's
import graph + make the dependency direction unclear. ``utils``
is meant to be a leaf (no imports from local_ai_platform.* other
modules) so it can safely be imported from anywhere.

Sources (2025-2026):
  * Wave 12 [IMPROVE-111] commit — initial creation of this
    package + the validation module.
  * Python ``inspect.signature`` introspection canonical
    reference (Python 3.11 docs):
    https://docs.python.org/3.11/library/inspect.html#inspect.signature
"""
