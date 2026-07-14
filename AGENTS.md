# AGENTS.md

## Purpose
Record practical engineering conventions learned in this repo, so future changes stay consistent and low-risk.

## Serving Architecture
- Keep `src/serving/server.py` as a thin service handler (request parsing, response mapping, error mapping, lifecycle hooks).
- Move orchestration/business execution logic out of handler files (for example into `src/serving/pipeline_runtime.py`).
- Do not mix route/controller concerns with pipeline node implementation details.

## Pipeline Orchestration
- Prefer dynamic pipeline composition via JSON `steps` arrays.
- Runtime should execute by looping steps in order and dispatching by a whitelisted node registry.
- Avoid hard-coded stage chains like `retriever -> adaptor -> calculator -> model -> executor` in Python control flow.
- New pipeline creation should require JSON-only changes whenever possible.
- New node creation should follow: implement node function/class -> register in whitelist -> reference `type` in JSON.

## Test Reliability
- Avoid network side effects at test import time.
- Integration-only initialization (for example exchange/client bootstrap) must be gated by `RUN_INTEGRATION_TESTS`.
- Keep default unit-test mode offline and deterministic.

## unittest Script Conventions
- In bash arrays, never place plain text notes as array entries (use comment lines with `# ...`).
- `run_unittest.sh` should fail on real test failures and also check runtime logs for unexpected errors.
- Log risk check should support allowlist patterns for expected error-path tests.
- Provide an explicit escape hatch (`--skip-log-check`) for special debugging sessions.

## Log Risk Policy
- Presence of `ERROR`/`CRITICAL` in test logs is a risk signal, but not always a failure.
- Treat only non-allowlisted errors as build-breaking.
- Keep allowlist minimal and explicit; add entries only for intentionally tested error paths.

## Backward Compatibility
- When evolving calculator APIs, keep compatibility with existing tests/callers (for example `batch_size` and `num_features` aliases), then migrate gradually.
- Preserve existing external API behavior (`/predict` compatibility fields and response shape) during refactors.

## Feature Contract Lessons
- Treat model input dimension as a hard contract (`dragonnet` currently expects 111).
- Do not silently remove feature columns that are part of trained input schema (including rank-like columns) unless model is retrained/migrated together.
- Keep zero-padding only as a safety fallback for serving availability, not as a primary mechanism.
- When `Generated feature dim < target dim` appears, treat it as a regression signal and investigate selection/exclusion changes first.

## 502/503 Incident Playbook
- Symptom: `/predict` returns 502 and details include TorchServe 503.
- First check TorchServe worker logs for shape errors (e.g. `mat1 and mat2 shapes cannot be multiplied`).
- Typical root cause: calculator output dim mismatch with model first linear layer input dim.
- Fix priority:
1. Restore correct feature set/dimension generation.
2. Keep fallback guards for clearer diagnostics.
3. Re-run `./run_unittest.sh` and `./test_deployment.sh` before merge.
