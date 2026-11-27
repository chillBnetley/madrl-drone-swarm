# Contributing

Thanks for looking. This repository accompanies a final-year project, so the useful
contributions are small, testable ones: a better world, a missing baseline, a metric, a
documentation fix for a pitfall you actually hit.

## Ways to help that do not need ROS 2 or Gazebo

Everything under `rl/` and `tests/` runs with **numpy only** (plus
`stable-baselines3`/`torch` if you want to train):

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pytest
python rl/check_env.py
python -m pytest tests -q
```

Good first issues are labelled
[`good first issue`](https://github.com/momen1231/madrl-drone-swarm/labels/good%20first%20issue);
they are scoped to be finishable in an evening without the full stack.

## Setting up the full stack

ROS 2 Humble + Gazebo Sim + PX4 SITL is a real install. Follow
[`docs/installation.md`](docs/installation.md), including the four documented pitfalls;
if one of them is wrong on your machine, that is a legitimate bug report.

## Before you open a pull request

1. `python -m pytest tests -q` passes.
2. `python rl/check_env.py` still prints `env OK`.
3. If you touched the simulation, paste the commands you ran and what you observed.
   "It builds" is not enough — say which topic, which pose or which error changed.
4. If you changed a reward term or an observation, say what it does to the results in
   the README table. Changing behaviour without touching the numbers makes them wrong.

## Style

* Python: 4 spaces, type hints where they help, `from __future__ import annotations`,
  no formatter config imposed yet — keep the diff readable.
* One logical change per pull request. A pull request that fixes a typo and rewrites a
  training loop cannot be reviewed.
* Commit messages: imperative subject line, then *why* in the body. If you used an
  AI assistant, say so in the pull request description.

## Reporting results

Numbers are welcome as issues or pull requests, with: algorithm, seed(s), timesteps,
the config you used (`rl/configs/*.yaml`), and the success criterion. A single-seed
number is accepted if it is labelled as such.

## What is out of scope

* Large refactors of the three-layer split (`sim/`, `ros2_ws/`, `rl/`) — it is
  deliberate.
* Non-ROS flight stacks (ArduPilot, etc.) — a separate project would be cleaner.
* Anything that requires a GPU cluster to evaluate.

## Licence

By contributing you agree that your work is released under the MIT licence of the
repository.
