# Third-Party Source

This directory is reserved for explicitly vendored third-party source code.

Current state: no third-party source is vendored.

Rules:

- every component must be listed in `MANIFEST.json`;
- every component must keep upstream `LICENSE` and `NOTICE` files when present;
- every component must include `UPSTREAM.md` and `PATCHES.md`;
- forbidden licenses must not appear in `third_party/permissive`;
- large runtime dependencies must remain package dependencies, not vendored source.
