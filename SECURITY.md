# Security Policy

## Supported Versions

The active development branch is the supported security target until the project publishes versioned
release lines.

## Reporting

Do not open a public issue for a vulnerability that can expose users or infrastructure. Report privately to
the project maintainer with:

- affected version or commit;
- reproduction steps;
- dependency and platform context;
- expected impact;
- any known workaround.

## Dependency And Vendor Rules

Large third-party runtimes must stay dependency-managed. Vendored code must pass `dtflowcv vendor-check`
and `dtflowcv license-check` before merge.
