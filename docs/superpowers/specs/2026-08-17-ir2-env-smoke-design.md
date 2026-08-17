# IR² Environment Smoke-Test Design

## Goal

Create an isolated, reproducible way to verify that the public IR² repository can load its pretrained Stage-2 policy and execute one inference episode using the repository's own map/environment code.

## Scope

- Work only on branch `chatgpt/ir2-env-smoke`; do not modify `master`.
- Use the repository's existing `model/stage2/checkpoint.pth` and `DungeonMaps/test` assets.
- Start with one `hybrid` map and one inference episode.
- Disable GIF generation and graph visualization for the smoke run so rendering does not dominate runtime.
- Avoid changing planner behavior in this setup task.

## Approach

1. Add a small smoke runner that directly instantiates the existing policy/environment/worker stack without Ray orchestration.
2. Configure the runner for CPU, one episode, hybrid map, no GIFs, and greedy inference.
3. Add a GitHub Actions workflow that checks out the full repository, creates a Python 3.8-compatible environment, installs the repository's documented dependency versions (with only minimal compatibility adjustments if package availability requires them), and runs the smoke runner.
4. Treat the smoke test as successful only when the Stage-2 checkpoint loads and an episode reaches the worker loop without infrastructure errors. Record the episode metrics even if the learned policy itself does not reach 99% exploration within its step limit.

## Data flow

`checkpoint.pth` → `PolicyNet` → `TestWorker`/`Env` → hierarchical graph + observations → greedy next-node actions → episode metrics.

## Diagnostics

The smoke runner should print:

- Python and key package versions
- checkpoint path and successful load confirmation
- selected test map and robot count
- elapsed wall time
- exploration rate, success flag, step count, and max travel distance
- a clear traceback/non-zero exit code for setup or infrastructure failures

## Follow-up

After the smoke run passes, use the same branch to add timing instrumentation for `update_graph`, observation construction, policy inference, environment stepping, and plotting, then repeat on a `complex` map.
