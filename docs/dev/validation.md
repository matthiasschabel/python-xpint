# Validation and regression evidence

**Status:** Active
**Last updated:** 2026-10-20
**Scope:** Release validation results, the language-neutral conformance contract, and benchmark procedure.

## Context

Tests are the behavioral evidence; this note records what they cannot: which
configurations a release was validated on, the shared conformance contract, and
how performance is measured. Results are dated and describe one commit. Rerun
them on the actual release commit; commands live in the
[release checklist](release_checklist.md).

## Current Decision

### Release validation

| Release | Commit | Platform / Python | Pint / NumPy | Result |
| --- | --- | --- | --- | --- |
| 0.1.0b1 | `350e0d6` (pre-publication history) | macOS arm64 / 3.12.12 | 0.25.3 / 2.5.2 | Installed wheel: 834 passed, 0 skipped, 91.77% coverage; strict mypy, Twine, minimal install passed |
| 0.1.0b1 | `350e0d6` | macOS arm64 / 3.13.14 | 0.25.3 / 2.5.2 | Same results |
| 0.1.0b1 | `350e0d6` | macOS arm64 / 3.14.6 (free-threaded build) | 0.25.3 / 2.5.2 | Same results |
| 0.1.0b1 | `a23e0f1` (tag `v0.1.0b1`) | hosted Ubuntu 24.04 / 3.12.15, 3.13.16, 3.14.8 | 0.25.3 / locked | `publish.yml` run 38838972361, installed wheel: 837 passed per job; minimal install passed |
| 0.1.0b1 | PyPI upload | macOS arm64 / 3.12.12 | 0.25.3 / — | Fresh `pip install xpint==0.1.0b1` from PyPI; minimal install passed |
| 0.1.0b2 | `44406f4` | macOS arm64 / 3.12.12 | 0.25.3 / 2.5.2 | Installed wheel: 843 passed, 0 skipped, 92% coverage; strict mypy, Twine, minimal install passed |
| 0.1.0b2 | `44406f4` | macOS arm64 / 3.13.14 | 0.25.3 / 2.5.2 | Same results |
| 0.1.0b2 | `44406f4` | macOS arm64 / 3.14.6 (free-threaded build) | 0.25.3 / 2.5.2 | Same results |
| 0.1.0b3 | `2e7ee0f` | macOS arm64 / 3.12.12 | 0.25.3 / 2.5.2 | Installed wheel: 871 passed, 0 skipped, 91.91% coverage; strict mypy, Twine, minimal install passed |
| 0.1.0b3 | `2e7ee0f` | macOS arm64 / 3.13.14 | 0.25.3 / 2.5.2 | Same results |
| 0.1.0b3 | `2e7ee0f` | macOS arm64 / 3.14.6 (free-threaded build, GIL disabled) | 0.25.3 / 2.5.2 | Same results |
| 0.1.0b3 | `193dec0` | hosted Ubuntu / 3.12, 3.13, 3.14 | 0.25.3 / locked | `ci.yml` run 39309970283: 871 passed per job |
| 0.1.0b3 | `193dec0` (tag `v0.1.0b3`) | hosted Ubuntu / 3.12.15, 3.13.16, 3.14.8 | 0.25.3 / locked | `publish.yml` run 39311796932, installed wheel: 871 passed per job; minimal install passed |
| 0.1.0b3 | PyPI upload | macOS arm64 / 3.12.12, 3.13.14, 3.14.6 (free-threaded) | 0.25.3 / — | Fresh no-cache `pip install xpint==0.1.0b3` from PyPI outside any checkout; minimal install passed |
| 0.1.0b4 | `cbb05c3` | macOS arm64 / 3.12.12 | 0.25.3 / 2.5.2 | Installed wheel: 1,026 passed, 0 skipped, 92.38% coverage; strict mypy, Twine, minimal install passed |
| 0.1.0b4 | `cbb05c3` | macOS arm64 / 3.13.14 | 0.25.3 / 2.5.2 | Same results |
| 0.1.0b4 | `cbb05c3` | macOS arm64 / 3.14.6 (free-threaded build, GIL disabled) | 0.25.3 / 2.5.2 | Same results |
| 0.1.0b4 | `a680e79` | hosted Ubuntu / 3.12, 3.13, 3.14 | 0.25.3 / locked | `ci.yml` run 39379428932: 1,026 passed per job |
| 0.1.0b4 | `a680e79` (tag `v0.1.0b4`) | hosted Ubuntu / 3.12.15, 3.13.16, 3.14.8 | 0.25.3 / locked | `publish.yml` run 39379661679, installed wheel: 1,026 passed per job; minimal install passed |
| 0.1.0b4 | PyPI upload | macOS arm64 / 3.12.12, 3.13.14, 3.14.6 (free-threaded) | 0.25.3 / — | Fresh no-cache `pip install xpint==0.1.0b4` from PyPI outside any checkout; minimal install passed |
| 0.1.0b5 | `95c8613` | macOS arm64 / 3.12.12 | 0.25.3 / 2.5.2 | Installed wheel: 1,077 passed, 0 skipped, 92.51% coverage; strict mypy, Twine, minimal install passed |
| 0.1.0b5 | `95c8613` | macOS arm64 / 3.13.14 | 0.25.3 / 2.5.2 | Same results |
| 0.1.0b5 | `95c8613` | macOS arm64 / 3.14.6 (free-threaded build, GIL disabled) | 0.25.3 / 2.5.2 | Same results |

The three local 0.1.0b5 builds of `95c8613` produced identical artifacts:
`xpint-0.1.0b5-py3-none-any.whl` SHA-256 `062cea4a3e64c3e9889dc3a6094148a1174ddd269d06d1bace19aa1c6fa531ba`,
`xpint-0.1.0b5.tar.gz` SHA-256 `f2206eb2bee8126a5f425b1e1d7cfa924f5e0d1f80e397effd62a00be0486e21`.

The three local 0.1.0b4 builds of `cbb05c3` produced identical artifacts:
`xpint-0.1.0b4-py3-none-any.whl` SHA-256 `5d54d063665d67c18787b09ad38de7a3d66f9154b816aa29abfd820fbf898f03`,
`xpint-0.1.0b4.tar.gz` SHA-256 `09d70f6b6f00bb90d5b7bd64f315871aeabce11b8828469f873dd697c99c597c`.
`a680e79` differs from `cbb05c3` only in maintainer docs, which neither artifact
contains; `publish.yml` run 39379661679 built the same two hashes from the tag.
Published 2026-10-20 06:22 UTC from that run; PyPI's SHA-256 digests match.

The three local 0.1.0b3 builds of `2e7ee0f` produced identical artifacts:
`xpint-0.1.0b3-py3-none-any.whl` SHA-256 `92ad77e2f324ba02c855bc8cd1f416bd992cf26a5aab3c3c40959326639005a0`,
`xpint-0.1.0b3.tar.gz` SHA-256 `9e8618a419aa3aab68d67a714678df81edd79044429c90a36e737ad740d32ec7`.
`193dec0` differs from `2e7ee0f` only in maintainer docs, which neither artifact
contains; `publish.yml` run 39311796932 built the same two hashes from the tag.
Published 2026-10-19 19:51 UTC from that run; PyPI's SHA-256 digests match.

The three local 0.1.0b2 builds of `44406f4` produced identical artifacts:
`xpint-0.1.0b2-py3-none-any.whl` SHA-256 `b6ca5818f44a67859225fa64e4c5b371476e723705f2039961510f886eb472d0`,
`xpint-0.1.0b2.tar.gz` SHA-256 `65a4dfdc9b3b7d14d3add92e69630bed246eb85ae7c24b40491a57481cfd97a3`.

The three local 0.1.0b1 builds produced identical artifacts:
`xpint-0.1.0b1-py3-none-any.whl` SHA-256 `79d96702ada720c746caa986ad81aa83e82738cee60b98dfbdf2125441deac78`,
`xpint-0.1.0b1.tar.gz` SHA-256 `c72b60b015dd942cce3a5004d34448588909f522aede3b71089b29e3747d3af3`.
The public root commit differs from `350e0d6` only by dropping agent guidance,
which is not packaged, and by `.gitignore`, which the sdist includes, so its
sdist hash will differ. The uploaded artifacts are the ones built by
`publish.yml` and are checked against that run's manifest.

Published 2026-10-16 02:13 UTC from run 38838972361. PyPI SHA-256 digests match
that run's manifest: wheel `851a93c97bb8952c551f0251471a96b2139b41e907bb17d7f8ef79078b1bf487`,
sdist `9fbd7806077dc6eb85bf3ef054d046217c712bc046ab186ddcf8badd32ee28ea`. The wheel's
file contents equal a local build of the tag; the sdist files equal the tagged tree.

Record one row per configuration once the release commit is validated: test
count, coverage, and whether the run used the installed wheel
(`XPINT_REQUIRE_WHEEL=1`) or the source tree. Local macOS results and hosted
Linux CI are separate evidence; neither substitutes for the other.

What the release checks establish:

- The wheel contains `xpint/`, `py.typed`, and the license; the sdist adds the
  examples and CHANGELOG. Tests, fixtures, maintainer docs, and agent guidance are
  excluded from both.
- Pint is the only runtime dependency. `tools/check_minimal_install.py` verifies
  that optional packages are absent and refuses to run under `python -O`, which
  would strip its assertions.
- The development dependency group installs every optional dependency the suite
  uses. A few tests still use `pytest.importorskip` (xarray, pint-xarray), so check
  the skip count in release runs; it should be zero.
- Inside pytest, `XPINT_REQUIRE_WHEEL=1` asserts that `xpint` was imported from
  site-packages, so a path override cannot test the source tree by accident.

### Language-neutral conformance

[semantic_conformance.json](../../tests/fixtures/semantic_conformance.json) is
schema version 1 with scalar cases. Its [Python consumer](../../tests/test_conformance.py)
uses public APIs and the default (dimensionless-angle) registry. No MATLAB
consumer has run, so the fixture is a contract candidate, not evidence of parity.
Parse cases hold only behavior meant to be shared: registry designation slots
(`mM[Fe]`, prefixed molar units) are xpint-specific and tested in the Python
suite instead, as is plain-target designation adoption.

- Operands give a magnitude, a unit name, an optional kind, and an optional
  factor designation. Unit names may map to each library's spelling.
- Operations are `add`, `subtract`, `multiply`, and `divide`; consumers map them
  to native APIs without evaluating source text.
- A successful expectation gives the reporting unit, magnitude, kind,
  temperature role, and sorted designation signature. Signature rows hold a
  physical dimension name, an opaque label, and a rational exponent as text; a
  volume label contributes exponent 3 to length. Conversion to the reporting
  unit must preserve the signature.
- The error category `semantic_mismatch` means rejection, not a sentinel value;
  each consumer maps it to its own exception type.
- Both tolerances are 1e-12, sized for scale and offset rounding in these short
  expressions, not as a general scientific tolerance.

Extend the cases only together with the other consumer's owner.

### Performance

```bash
uv run python benchmarks/benchmark_pint_passthrough.py
uv run python benchmarks/benchmark_semantic_safety.py --backend pint
uv run python benchmarks/benchmark_semantic_safety.py --backend xpint
uv run python benchmarks/benchmark_semantic_safety.py --source-root /path/to/other/src
```

Run source variants in fresh processes, rotate their order, and record Python,
Pint, NumPy, hardware, and source revisions. The safety harness writes JSON to
stdout and diagnostics to stderr, and reports timings and traced allocations
separately. Compare equal dtype, alias, shape, and failure contracts. Ratios are
diagnostics, not CI assertions.

Indicative ratios to Pint on 1,000-element arrays and scalars (Python 3.12, Pint
0.25.3, NumPy 2.5.2, Apple silicon, 2026-10-15): unclassified arithmetic about
1.4–1.5×, kinded addition about 1.5×, kinded multiplication about 3×, `to()`
about 1.35×. The cost is the classification checks and result construction.
Staged in-place NumPy operations (powers, `ito()`, temperature-role addition and
subtraction, classified multiplication and division) also copy the buffer so
that a failure leaves values and metadata unchanged; that copy grows with array
size and is deliberate. Temperature-role `+=` and `-=` commit their computed
result directly when its shape and dtype match a writable destination, so they
hold one temporary buffer rather than two (at 1M float64 elements: 0.81 ms
versus 1.34 ms before, Pint 0.17 ms).

## Alternatives Considered

Editable-install tests do not establish wheel correctness. Silently skipped
optional tests hide adapter gaps. Microbenchmark ratios without controlled
variance cannot support performance-neutrality claims.

## Deferred Work

Hosted Windows CI, a broader dependency matrix, and the MATLAB conformance
consumer.

## Next Steps

Fill in the release row from the validated release commit. Keep changed
contracts in [architecture](../architecture.md) and
[compatibility](../compatibility.md), and public behavior changes in the root
CHANGELOG.
