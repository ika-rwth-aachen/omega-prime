# Third-Party Licenses

omega-prime is distributed under the [Mozilla Public License 2.0](LICENSE).

Dependency licenses are checked automatically against MPL-2.0 by
[`licensecheck`](https://github.com/FHPythonUtils/LicenseCheck), configured under
`[tool.licensecheck]` in [`pyproject.toml`](pyproject.toml) and enforced by
`.github/workflows/licensecheck.yml` (push to `main`, and every pull request) as
well as by the `licensecheck` job in `.gitlab-ci.yml`.

To reproduce the check locally:

```bash
uvx licensecheck==2026.0.8 --zero --format ansi
```

Only *runtime* dependencies are in scope. Test and documentation tooling is
declared in the `test` extra, is never redistributed as part of omega-prime, and
therefore carries no license obligation for downstream users.

This file records the deliberate exceptions configured in that check, so the
reasoning behind them is not lost.

## Apache-2.0 dependencies — accepted by policy

`ignore_licenses = ["MPL", "Apache"]`

licensecheck's built-in compatibility matrix marks Apache-2.0 as incompatible
with an MPL-2.0 project. That is over-conservative: Apache-2.0 is a permissive
license, and MPL-2.0 §3.3 explicitly allows Covered Software to be distributed
as part of a Larger Work under other terms. Apache-2.0 dependencies are
therefore accepted as a matter of policy rather than being enumerated one by one
— an allowlist of individual packages has to be extended every time a new
transitive Apache-2.0 dependency appears.

Currently reached this way: `pyarrow`, `grpcio`, `xarray`, `multidict`.

## `polars-st` — LGPL-2.1

PyPI reports `polars-st` as LGPL-2.1, because its wheels statically link
[GEOS](https://libgeos.org/), which is LGPL-2.1.

**Accepted for the current distribution model.** omega-prime is published as a
pure-Python wheel that declares `polars-st` as a dependency and imports it at
runtime. It neither contains nor redistributes any GEOS code, so the LGPL's
distribution obligations fall on whoever distributes `polars-st` itself, and
LGPL-2.1 places no license requirement on the calling work.

> **Before vendoring or freezing a build, revisit this.** If omega-prime is ever
> shipped as a bundle that embeds its dependencies — a PyInstaller/cx_Freeze
> executable, a container image with vendored wheels, or any single-file
> distribution — LGPL-2.1 §6 relinking obligations attach to that artifact:
> recipients must be able to replace the LGPL component with a modified version.

## `mypy-extensions` — missing license metadata

Reached transitively via `pandera` → `typing-inspect`.

`mypy-extensions` publishes no `license`, `license_expression`, or license
classifier to PyPI, so licensecheck cannot classify it and fails it as unknown.
The package is in fact MIT-licensed — see
[python/mypy_extensions](https://github.com/python/mypy_extensions). It is
ignored on that basis.
