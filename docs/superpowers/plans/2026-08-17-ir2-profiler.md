# IR² Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable profiler for one deterministic IR² inference episode and use it to identify the dominant runtime functions without changing exploration behavior.

**Architecture:** Keep the existing `chatgpt_smoke_test.py` baseline unchanged. Add a separate `profile_ir2.py` that reuses the same deterministic hybrid-map setup, profiles only `worker.work(0)` with Python `cProfile`, saves a `.prof` file, and prints top cumulative-time and self-time functions. Extend GitHub Actions with a dedicated profile job so the full repository checkpoint/maps are used in the authoritative run.

**Tech Stack:** Python 3.8, `cProfile`, `pstats`, PyTorch 1.10.0, existing IR² `TestWorker`/`Env`, GitHub Actions.

## Global Constraints

- Work only on branch `chatgpt/ir2-env-smoke`; do not modify `master`.
- Use the repository's existing `model/stage2/checkpoint.pth` and `DungeonMaps/test` assets.
- Use the same deterministic `hybrid` episode configuration as `chatgpt_smoke_test.py`.
- Keep GIF generation and graph visualization disabled.
- Do not modify planner behavior, `env.py`, `test_multi_robot_worker.py`, or model logic for profiling.
- Report both cumulative time and self time because nested calls overlap.

---

### Task 1: Add the reusable profiler utility

**Files:**
- Create: `profile_ir2.py`
- Create: `tests/test_profile_ir2.py`

**Interfaces:**
- Consumes: `chatgpt_smoke_test.configure_smoke_parameters`, `chatgpt_smoke_test.load_policy`
- Produces: `profile_callable(callable_obj, *args, dump_path=None, **kwargs)` returning `(result, pstats.Stats)`; CLI entry point that profiles one IR² episode.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from profile_ir2 import profile_callable


def test_profile_callable_returns_result_and_writes_profile(tmp_path):
    output = tmp_path / "sample.prof"

    def workload():
        return sum(range(100))

    result, stats = profile_callable(workload, dump_path=output)

    assert result == 4950
    assert output.is_file()
    assert stats.total_calls > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_profile_ir2.py -q`
Expected: FAIL because `profile_ir2` does not exist.

- [ ] **Step 3: Implement the minimal profiler utility and IR² CLI**

`profile_ir2.py` must:

```python
import cProfile
import pstats
import time
from pathlib import Path


def profile_callable(callable_obj, *args, dump_path=None, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = callable_obj(*args, **kwargs)
    finally:
        profiler.disable()
    if dump_path is not None:
        profiler.dump_stats(str(dump_path))
    return result, pstats.Stats(profiler)
```

The CLI must reproduce the same seeds, CPU device, 4 robots, `global_step=0`, greedy policy, and no rendering from the smoke runner; profile only `worker.work(0)`; save `ir2_profile.prof`; print wall time and episode metrics; then print top 40 entries sorted by `cumulative` and top 40 sorted by `tottime`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_profile_ir2.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Run syntax/import checks**

Run: `python -m py_compile profile_ir2.py`
Expected: exit code 0.

### Task 2: Run the profiler on the real repository assets

**Files:**
- Modify: `.github/workflows/ir2-smoke.yml`

**Interfaces:**
- Consumes: `profile_ir2.py`, existing Python 3.8 dependency environment, pretrained checkpoint and hybrid maps.
- Produces: GitHub Actions logs containing profile tables and `ir2_profile.prof` on the runner.

- [ ] **Step 1: Extend the existing workflow**

Add after the smoke episode:

```yaml
      - name: Run IR2 profiler
        run: python profile_ir2.py
```

Keep the existing smoke test and baseline episode so profiler results can be checked against a known-good run.

- [ ] **Step 2: Push the workflow change and wait for Actions**

Expected: checkout, Python setup, dependency install, smoke unit tests, baseline inference, and profiler all complete successfully.

- [ ] **Step 3: Read profiler output**

Extract:
- profiled wall time and episode metrics;
- top cumulative-time functions;
- top self-time functions;
- call counts for the dominant functions.

- [ ] **Step 4: Identify the bottleneck from evidence**

State the dominant function/category only if both the profile data and call structure support it. Separate inclusive/cumulative time from self time, and do not sum overlapping cumulative times.

- [ ] **Step 5: Compare against baseline**

Confirm the profiled run uses the same map/robot configuration and achieves comparable episode metrics; note cProfile overhead, so use percentages/order to diagnose hotspots rather than treating profiled wall time as the uninstrumented runtime.
