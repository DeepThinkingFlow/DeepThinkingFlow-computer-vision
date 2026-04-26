# Vendor Policy

Vendoring is allowed only for small, stable, license-compatible components with a clear reason.

Allowed by default:

- Apache-2.0
- MIT
- BSD-2-Clause
- BSD-3-Clause
- ISC
- Zlib
- 0BSD
- CC0-1.0 for data/config assets, not substantial code

Requires explicit review:

- MPL-2.0
- LGPL-2.1
- LGPL-3.0
- EPL-2.0
- CDDL

Forbidden in the Apache core tree:

- GPL-2.0
- GPL-3.0
- AGPL-3.0
- SSPL
- custom non-commercial licenses
- research-only licenses
- unknown or missing licenses

Every vendored component must include:

- `LICENSE`
- `NOTICE` if upstream provides one
- `UPSTREAM.md`
- `PATCHES.md`
- an entry in `third_party/MANIFEST.json`

Do not vendor Ultralytics, Torch, OpenCV, ONNXRuntime, NumPy, Pandas, Matplotlib, or MLflow into this
repository. Keep those as dependency-managed optional integrations.
