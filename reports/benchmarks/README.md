# Benchmark Artifacts

This directory is for committed benchmark evidence, not ad hoc local runs.

Required fields for accepted model-quality evidence:

- `benchmark_id`
- `git_commit`
- `model_sha256`
- `dataset_sha256`
- `class_schema_sha256`
- `image_count`
- `metric.ap50`
- `metric.ap50_95`
- `metric.ap75`
- `latency.end_to_end_latency_ms`
- `memory.max_rss_mb`
- `dependency_versions`
- `claim_boundary`

Current boundary: this repo has smoke and evaluator contracts, but no accepted
real model-quality benchmark artifact is committed yet.
