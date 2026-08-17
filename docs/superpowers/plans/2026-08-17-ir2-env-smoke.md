# IR² Environment Smoke Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible CPU-only smoke path that loads the repository's Stage-2 checkpoint and runs one hybrid-map IR² inference episode without Ray or visualization.

**Architecture:** Keep the existing IR² planner, environment, graph code, and trained weights unchanged. Add a small wrapper that patches only inference configuration before importing `TestWorker`, then loads `PolicyNet` and executes one greedy episode directly. A GitHub Actions workflow provides the Python 3.8 environment and runs both a small configuration unit test and the full integration smoke episode.

**Tech Stack:** Python 3.8, PyTorch 1.10.0, NumPy 1.23.5, scikit-image 0.19.3, scikit-learn 1.2.1, SciPy 1.10.0, Matplotlib 3.6.3, imageio, pytest, GitHub Actions.

## Global Constraints

- Work only on branch `chatgpt/ir2-env-smoke`; do not modify `master`.
- Use `model/stage2/checkpoint.pth` and `DungeonMaps/test/hybrid` already present in the repository.
- Run exactly one hybrid-map inference episode on CPU with four robots and greedy actions.
- Disable GIF generation and graph visualization.
- Do not change planner, reward, graph, motion, communication, or termination behavior.
- A policy failure to reach 99% is a reported metric; infrastructure/A* setup failure is a smoke-test failure.

---

### Task 1: Direct smoke runner

**Files:**
- Create: `chatgpt_smoke_test.py`
- Create: `tests/test_chatgpt_smoke_test.py`

**Interfaces:**
- Produces: `configure_smoke_parameters()` which mutates the existing `test_parameter` module before `TestWorker` is imported.
- Produces: `load_policy(device)` which loads `model/stage2/checkpoint.pth` into `PolicyNet`.
- Produces: `run_smoke_episode()` which returns the worker metrics dictionary after one episode, or raises on infrastructure failure.

- [ ] **Step 1: Write the configuration test**

```python
from chatgpt_smoke_test import configure_smoke_parameters


def test_smoke_configuration_is_cpu_hybrid_without_rendering():
    cfg = configure_smoke_parameters()
    assert cfg.TEST_SET_NAME == "hybrid"
    assert cfg.TEST_SET_DIR == "DungeonMaps/test/hybrid"
    assert cfg.NUM_ROBOTS_MIN == cfg.NUM_ROBOTS_MAX == 4
    assert cfg.SAVE_GIFS is False
    assert cfg.VIZ_GRAPH_EDGES is False
    assert cfg.VIZ_GRAPH_EDGES_GROUND_TRUTH is False
    assert cfg.USE_GPU is False
```

- [ ] **Step 2: Run test to verify it fails before the runner exists**

Run: `pytest tests/test_chatgpt_smoke_test.py -q`
Expected: import failure for `chatgpt_smoke_test`.

- [ ] **Step 3: Implement `chatgpt_smoke_test.py`**

The runner must:

```python
import importlib.metadata
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

CHECKPOINT = Path("model/stage2/checkpoint.pth")


def configure_smoke_parameters():
    import test_parameter as cfg
    cfg.TEST_SET_NAME = "hybrid"
    cfg.TEST_SET_DIR = "DungeonMaps/test/hybrid"
    cfg.MAX_EPS_STEPS = 196
    cfg.K_SIZE = 30
    cfg.NUM_ROBOTS_MIN = 4
    cfg.NUM_ROBOTS_MAX = 4
    cfg.NODE_COORDS_SCALING_FACTOR = 1 / 640
    cfg.GLOBAL_GRAPH_NODE_COORDS_THRESH = 200
    cfg.SAVE_GIFS = False
    cfg.VIZ_GRAPH_EDGES = False
    cfg.VIZ_GRAPH_EDGES_GROUND_TRUTH = False
    cfg.USE_GPU = False
    cfg.NUM_META_AGENT = 1
    return cfg


def load_policy(device):
    cfg = configure_smoke_parameters()
    from model import PolicyNet
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(str(CHECKPOINT))
    policy = PolicyNet(cfg.INPUT_DIM, cfg.EMBEDDING_DIM).to(device)
    checkpoint = torch.load(str(CHECKPOINT), map_location=device)
    policy.load_state_dict(checkpoint["policy_model"])
    policy.eval()
    return policy
```

`run_smoke_episode()` must set `random`, NumPy, and Torch seeds to 0, import `TestWorker` only after configuration, instantiate it with `meta_agent_id=0`, `n_agent=4`, `global_step=0`, `device=torch.device("cpu")`, `greedy=True`, `save_image=False`, and call `worker.work(0)`. If `worker.work(0)` returns `False`, raise `RuntimeError("IR2 worker reported infrastructure/A* failure")`; otherwise return `worker.perf_metrics` and print versions, selected map, elapsed time, explored rate, success flag, steps, and travel distance.

- [ ] **Step 4: Run the unit test**

Run: `pytest tests/test_chatgpt_smoke_test.py -q`
Expected: PASS.

- [ ] **Step 5: Syntax-check the integration runner**

Run: `python -m py_compile chatgpt_smoke_test.py`
Expected: exit 0.

- [ ] **Step 6: Commit runner and test**

```bash
git add chatgpt_smoke_test.py tests/test_chatgpt_smoke_test.py
git commit -m "test: add direct IR2 inference smoke runner"
```

---

### Task 2: Reproducible GitHub Actions environment

**Files:**
- Create: `.github/workflows/ir2-smoke.yml`

**Interfaces:**
- Consumes: `chatgpt_smoke_test.py` and `tests/test_chatgpt_smoke_test.py`.
- Produces: a branch-scoped workflow run whose logs are the evidence for environment setup and full inference execution.

- [ ] **Step 1: Add the workflow**

Use `ubuntu-22.04`, `actions/checkout@v4`, and `actions/setup-python@v5` with Python `3.8`. Set `MPLBACKEND=Agg` and `PYTHONUNBUFFERED=1`. Install:

```text
numpy==1.23.5
torch==1.10.0
scikit-image==0.19.3
scikit-learn==1.2.1
scipy==1.10.0
matplotlib==3.6.3
imageio==2.31.6
pytest==7.4.4
```

The workflow must trigger on `push` to `chatgpt/ir2-env-smoke` and on `workflow_dispatch`, have a 30-minute job timeout, run `pytest tests/test_chatgpt_smoke_test.py -q`, then run `python chatgpt_smoke_test.py`.

- [ ] **Step 2: Commit the workflow**

```bash
git add .github/workflows/ir2-smoke.yml
git commit -m "ci: run IR2 smoke inference on Python 3.8"
```

- [ ] **Step 3: Inspect the triggered workflow run**

Expected: dependency setup, unit test, and integration smoke steps appear in the run. If dependency installation or execution fails, use the exact job logs to identify the first failing component before changing anything.

- [ ] **Step 4: Apply one minimal compatibility correction at a time if required**

Examples of acceptable setup-only corrections are a package wheel version pin or CPU PyTorch index selection. Do not change IR² planning behavior to make the workflow pass.

- [ ] **Step 5: Re-run and verify**

Success criteria: the job exits 0; the log confirms checkpoint loading, identifies the hybrid map/four robots, and prints elapsed time plus `explored_rate`, `success_rate`, `travel_steps`, and `travel_dist`.

---

## Self-review

- Spec coverage: isolated branch, existing checkpoint/maps, hybrid/CPU/no-rendering, no Ray, diagnostics, and metrics are all covered.
- No planner source files are modified.
- Unit configuration test and full integration run exercise different failure layers.
- The workflow pins NumPy below 1.24 to retain compatibility with legacy NumPy APIs used by the repository while keeping the documented scientific-package versions.
