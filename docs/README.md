# Documentation map

User documentation lives in the root [README](../README.md), package docstrings,
and [examples](../examples/README.md). [overview.html](overview.html) is the authored
source of the separately published visual overview; editing it locally does not
update an existing published copy.

Maintainer documents are organized by purpose:

| Document | Use it for |
| --- | --- |
| [Architecture](architecture.md) | Objectives, implemented model, ownership, and design alternatives |
| [Compatibility](compatibility.md) | Current integration contracts, public test map, and Pint upgrade boundaries |
| [Future work](dev/future_work.md) | Unsupported capabilities, prerequisites, and unresolved design questions |
| [Upstream Pint](dev/upstream_pint.md) | Pending contributions and dependency adoption conditions |
| [Validation](dev/validation.md) | Release validation results, conformance contract, and benchmark procedure |
| [Release checklist](dev/release_checklist.md) | Release gates, publishing steps, and verification commands |

The `docs/` tree is excluded from wheel and sdist. The visual overview is published
separately; maintainer notes are not release payload. Tests/fixtures and benchmark
harnesses remain in their own top-level trees.

## Keeping the tree useful

Update the logical document instead of appending a new audit, review transcript,
patch bundle, or private changelog. Retain a separate plan only while the work is
active; promote enduring decisions when it completes, and remove superseded
material after preserving outstanding actions. Git history holds completed work.
Review transport logs belong outside tracked documentation.

QA fixes go in the root CHANGELOG when user-visible; update compatibility and
validation for the enduring evidence. Do not recreate a parallel
`docs/<namespace>/dev/changelog.md`. This project-specific convention implements
the requested consolidation even when a generic QA workflow suggests a separate
maintainer changelog. Current agent instructions belong in root AGENTS.md, not
these documents.
