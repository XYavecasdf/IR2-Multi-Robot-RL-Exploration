"""cProfile-based performance profiler for one deterministic IR2 episode."""

import cProfile
import pstats
import time
from pathlib import Path

PROFILE_PATH = Path("ir2_profile.prof")


def profile_callable(callable_obj, *args, dump_path=None, **kwargs):
    """Profile one callable and return both its result and pstats object."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = callable_obj(*args, **kwargs)
    finally:
        profiler.disable()

    if dump_path is not None:
        profiler.dump_stats(str(dump_path))

    return result, pstats.Stats(profiler)


def print_profile_table(stats, sort_key, limit=40):
    """Print one sorted cProfile table without changing program behavior."""
    stats.strip_dirs().sort_stats(sort_key).print_stats(limit)


def main():
    import random

    import numpy as np
    import torch

    from chatgpt_smoke_test import configure_smoke_parameters, load_policy

    cfg = configure_smoke_parameters()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cpu")
    policy = load_policy(device)

    # Import after inference globals are patched; TestWorker imports the
    # configuration with `from test_parameter import *`.
    from test_multi_robot_worker import TestWorker

    worker = TestWorker(
        meta_agent_id=0,
        n_agent=4,
        policy_net=policy,
        global_step=0,
        device=device,
        greedy=True,
        save_image=False,
    )

    print("=== IR2 profiler setup ===")
    print("Test set:", cfg.TEST_SET_NAME)
    print("Map:", worker.env.file_path)
    print("Robots:", worker.n_agent)
    print("Profile output:", PROFILE_PATH)

    started = time.perf_counter()
    infrastructure_ok, stats = profile_callable(
        worker.work,
        0,
        dump_path=PROFILE_PATH,
    )
    elapsed = time.perf_counter() - started

    if not infrastructure_ok:
        raise RuntimeError("IR2 worker reported infrastructure/A* failure")

    metrics = worker.perf_metrics
    print("=== IR2 profiler result ===")
    print("profiled_elapsed_seconds: {:.3f}".format(elapsed))
    print("explored_rate:", metrics["explored_rate"])
    print("success_rate:", metrics["success_rate"])
    print("travel_steps:", metrics["travel_steps"])
    print("travel_dist:", metrics["travel_dist"])

    print("=== TOP 40 BY CUMULATIVE TIME (inclusive) ===")
    print_profile_table(stats, "cumulative", 40)
    print("=== TOP 40 BY SELF TIME (tottime) ===")
    print_profile_table(stats, "tottime", 40)


if __name__ == "__main__":
    main()
