# Release checklist

**Status:** Active
**Last updated:** 2026-10-19
**Scope:** Publishing xpint releases to PyPI; currently 0.1.0b3.

## Context

Releases are published by `.github/workflows/publish.yml` through PyPI Trusted
Publishing: manual dispatch on a `v*` tag builds the wheel and sdist once,
verifies those exact files on Python 3.12–3.14, and uploads them after approval
of the `pypi` environment. No token is stored. Tagging or creating a GitHub
release never publishes.

Owners: **M** = maintainer (accounts, settings, visibility, approvals);
**I** = implementation (repository edits and commands). An item stays unchecked
until its evidence exists; record release results in [validation](validation.md).

## Current Decision

### Before tagging

- [x] **I** Release text: README and CHANGELOG describe the release without
  "unpublished" wording; the CHANGELOG date is the intended upload date. Changing
  it later means a new commit and repeated validation.
- [x] **I** Version agreement: `pyproject.toml`, `xpint.__version__`, the
  CHANGELOG heading, and the planned tag all name 0.1.0b3 / `v0.1.0b3`.
- [x] **I** Local installed-artifact validation on 3.12, 3.13, and 3.14 (commands
  below), recorded in validation.
- [ ] **M+I** Publication review of the public tree before visibility changes:
  no credentials, private repository names, local paths, personal data, or
  reviewer transcripts. The public repository starts from a single root commit;
  earlier development history is archived privately.
- [ ] **M** Public repository at `github.com/matthiasschabel/python-xpint`, with
  every `[project.urls]` destination loading while logged out.
- [ ] **I** Hosted `ci.yml` green on the release commit, all three Python jobs.
  Local results do not substitute for hosted CI.
- [ ] **M** GitHub environment `pypi`: maintainer as required reviewer, deployment
  limited to `v*` tags, no secrets.
- [ ] **M** PyPI pending publisher matches exactly: project `xpint`, owner
  `matthiasschabel`, repository `python-xpint`, workflow `publish.yml`,
  environment `pypi`.

### Tag, build, and upload

- [ ] **M** Push annotated tag `v0.1.0b3` on the green commit, then dispatch
  `gh workflow run publish.yml --ref v0.1.0b3`. The build and verify jobs pass and
  the run waits for `pypi` approval.
- [ ] **M+I** Review the run's artifacts before approving: exactly
  `xpint-0.1.0b3-py3-none-any.whl` and `xpint-0.1.0b3.tar.gz`, SHA-256 values
  matching the build log, wheel holding `xpint/`, `py.typed`, and the license,
  sdist adding examples and CHANGELOG, neither holding tests or docs.
  ```bash
  gh run download <run-id> -n dist -D <new-empty-dir>
  shasum -a 256 <new-empty-dir>/*
  python -m zipfile -l <new-empty-dir>/xpint-0.1.0b3-py3-none-any.whl
  tar -tzf <new-empty-dir>/xpint-0.1.0b3.tar.gz
  ```
- [ ] **M** Approve the `pypi` deployment, naming the run and both hashes.

### After upload

- [ ] **I** PyPI holds the approved bytes:
  ```bash
  curl -s https://pypi.org/pypi/xpint/0.1.0b3/json | python3 -c 'import json,sys
  for u in json.load(sys.stdin)["urls"]: print(u["digests"]["sha256"], u["filename"])'
  ```
- [ ] **I** Fresh install from PyPI per supported Python, outside any checkout,
  with no cache and no optional packages, then `tools/check_minimal_install.py`.
- [ ] **M** The PyPI page renders the README and its links work logged out.
- [ ] **M** Optional GitHub pre-release on `v0.1.0b3` with the CHANGELOG notes.
- [ ] **I** Record results in validation and reset this checklist for the next
  release.

### Failure recovery

- PyPI filenames are permanent, even after deletion. Never replace uploaded bytes
  with a rebuilt file.
- Nothing uploaded (publisher mismatch or name rejection): keep the error, fix the
  configuration, and retry with the retained, hash-checked artifacts rather than
  rebuilding.
- Partial upload: compare every PyPI filename and SHA-256 against the approved set
  before uploading the missing file; `skip-existing` alone proves nothing.
- Defective release: yank it with a reason and release the next pre-release
  through this checklist. Do not delete it or move the tag.

### Commands

Development bar: `uv sync --locked`, `uv run ruff check .`,
`uv run ruff format --check .`,
`uv run mypy src tests examples benchmarks tools/check_minimal_install.py`, and
`uv run pytest`.

Installed-artifact sequence, per Python version (set `UV_PYTHON`; use a separate
checkout per version; copy reviewed files into `dist/` instead of building when
checking release artifacts):

```bash
uv sync --locked --no-install-project
uv build --no-sources                       # omit when testing reviewed artifacts
uv pip install --no-deps dist/*.whl
XPINT_REQUIRE_WHEEL=1 uv run --no-sync pytest
uv run --no-sync mypy src tests examples benchmarks tools/check_minimal_install.py
uv run --no-sync twine check --strict dist/*
uv venv .venv-minimal
uv pip install --python .venv-minimal/bin/python dist/*.whl
.venv-minimal/bin/python tools/check_minimal_install.py
```

## Alternatives Considered

- An empty upload to hold the name: rejected; publish only a reviewed release.
- A local build uploaded with an API token: a new project would need an
  account-scoped token, while Trusted Publishing stores no secret.
- Publishing on tag push or GitHub release: tagging would publish before the
  artifacts are reviewed.
- Rebuilding in the publish job: it would upload bytes nobody tested.

## Deferred Work

Pint 0.26 support, Windows CI, a TestPyPI rehearsal, a tag ruleset protecting
`v*`, and PyPI attestations review.

## Next Steps

Work through "Before tagging" in order, then tag and dispatch.
