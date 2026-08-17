"""Minimal, Ray-free inference smoke runner for the public IR2 repository."""

import importlib.metadata
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

CHECKPOINT = Path("model/stage2/checkpoint.pth")


def configure_smoke_parameters():
    """Patch inference-only globals before importing TestWorker/Env."""
    import test_parameter as cfg

    cfg.TEST_SET_NAME = "hybrid"
    cfg.TEST_SET_DIR = "DungeonMaps/test/hybrid"
    cfg.MAX_EPS_STEPS = 196
    cfg.K_SIZE = 30
    cfg.NUM_ROBOTS_MIN = 4
    cfg.NUM_ROBOTS_MAX = 4
    cfg.NODE_COORDS_SCALING_FACTOR = 1 / 640
    cfg.NODE_UTILITY_SCALING_FACTOR = 1 / 50
    cfg.GLOBAL_GRAPH_NODE_COORDS_THRESH = 200
    cfg.NODE_PADDING_SIZE = 99999
    cfg.SAVE_GIFS = False
    cfg.VIZ_GRAPH_EDGES = False
    cfg.VIZ_GRAPH_EDGES_GROUND_TRUTH = False
    cfg.USE_GPU = False
    cfg.NUM_META_AGENT = 1
    return cfg


def load_policy(device):
    """Load the repository's pretrained Stage-2 policy on the requested device."""
    cfg = configure_smoke_parameters()
    from model import PolicyNet

    if not CHECKPOINT.is_file():
        raise FileNotFoundError("Missing pretrained checkpoint: {}".format(CHECKPOINT))

    policy = PolicyNet(cfg.INPUT_DIM, cfg.EMBEDDING_DIM).to(device)
    checkpoint = torch.load(str(CHECKPOINT), map_location=device)
    policy.load_state_dict(checkpoint["policy_model"])
    policy.eval()
    return policy


def _version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def print_environment():
    print("=== IR2 smoke environment ===")
    print("Python:", platform.python_version())
    for distribution in (
        "numpy",
        "torch",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "imageio",
    ):
        print("{}: {}".format(distribution, _version(distribution)))


def run_smoke_episode():
    """Execute one deterministic hybrid-map episode without Ray or rendering."""
    cfg = configure_smoke_parameters()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cpu")
    print("Checkpoint:", CHECKPOINT)
    policy = load_policy(device)
    print("Checkpoint load: OK")

    # Import only after the inference globals above have been patched because
    # TestWorker and Env use `from test_parameter import *` at module import time.
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
    print("Test set:", cfg.TEST_SET_NAME)
    print("Map:", worker.env.file_path)
    print("Robots:", worker.n_agent)

    started = time.perf_counter()
    infrastructure_ok = worker.work(0)
    elapsed = time.perf_counter() - started

    if not infrastructure_ok:
        raise RuntimeError("IR2 worker reported infrastructure/A* failure")

    metrics = worker.perf_metrics
    print("=== IR2 smoke result ===")
    print("elapsed_seconds: {:.3f}".format(elapsed))
    print("explored_rate:", metrics["explored_rate"])
    print("success_rate:", metrics["success_rate"])
    print("travel_steps:", metrics["travel_steps"])
    print("travel_dist:", metrics["travel_dist"])
    return metrics


def main():
    print_environment()
    run_smoke_episode()


if __name__ == "__main__":
    main()
