# Where the data comes from

There is no dataset. Every number this repository publishes was produced locally by running a
simulator, and every exported episode carries the seed that reproduces it. This page describes
the simulator, what it returns, its licence, and how far "produced locally" goes as a statement
of provenance.

This description was written from Gymnasium 1.0.0 on 2026-09-04, and `pyproject.toml` pins
that version.

## The environment

| | |
|---|---|
| Identifier | `LunarLander-v3` |
| Provider | [Gymnasium](https://gymnasium.farama.org/), Farama Foundation, version 1.0.0 |
| Physics | Box2D through `Box2D>=2.3.10` |
| Licence | Gymnasium: MIT. Box2D: zlib. Both permit use, modification and redistribution. |
| Options used | the defaults: wind off, hence no turbulence, and the default gravity |

A lander starts at the top of the frame with a random initial push and has to come to rest on a
pad between two flags. The environment does not model fuel, and every engine firing lowers the
reward.

The policy reads eight numbers and returns one of four actions.

| Position in the vector | Observation | Unit |
|---|---|---|
| first and second | position, horizontal and vertical | scaled by the environment, with the pad at the origin |
| third and fourth | velocity, horizontal and vertical | scaled by the environment |
| fifth and sixth | angle, and angular velocity | radians, and a scaled angular velocity |
| seventh and eighth | left and right leg in contact | a flag, zero or one |

The four actions are to do nothing, to fire the left engine, to fire the main engine and to fire
the right engine. The API in `src/rl_lander/api.py` rejects an observation outside the bounds
the environment declares. The policy returns an action for any eight numbers, including an
altitude of 900 that no episode can reach, so the check runs before the policy is called.

## What a seed fixes

`env.reset(seed=n)` fixes the initial state of an episode and the random draws made during it.
Two runs with the same seed, the same policy and the same package versions, on the same
platform, produce the same trajectory. `tests/system/test_service_end_to_end.py` asks the
running service for the same episode twice and checks that the two action sequences are equal.
`docs/operations.md` reports one episode that scores slightly differently on two platforms.

A seed fixes the training less completely. `set_global_seed` seeds Python's `random`, NumPy
and PyTorch, on the CPU and on every CUDA device, and it turns off the cuDNN autotuner. It does
not guarantee a training identical bit for bit. Its docstring notes that string hashing stays
random within the process unless `PYTHONHASHSEED` was set before the interpreter started, and
the DQN run trains on the GPU, whose kernels can be non-deterministic. The repository holds no
repeated training, so how close a retraining comes to the committed weights has not been
measured.

## The evaluation grids

An evaluation grid is a starting seed and a count. The grid that starts at seed g plays 100
episodes, reset with the seeds g, g + 1 and so on up to g + 99. Six grids are used, `2024, 7,
99, 123, 555, 31337`, and the episodes of the first one are exported row by row. Grids that
start close together share episodes: grids 99 and 123 both play the 76 episodes seeded 123 to
198, and grids 7 and 99 both play the 8 episodes seeded 99 to 106.

The six grids already appear in the evaluation manifest committed in 9c8fb3f, which records a
scoring of the previous policy on them at 05:47 UTC on 4 September 2026. The manifest committed
in 471b052 records the scoring of the shipped policy on the same grids at 08:34 UTC that day, so
the grids were in use before the shipped policy was scored. The git revisions that these two
manifests record are absent from the published history. The current manifest comes from a rerun
of `scripts/evaluate_and_export.py` on 27 October 2026, which wrote the episodes and the summary
again with identical content and records revision 15f576e of the published history. Of the six,
the exported grid has the lowest mean reward, the lowest landing rate and the lowest score on a
single episode.

## What is published, and what it contains

The table lists the files of results under `reports/`. Each is written by a script of this
repository from runs of the simulator, and all of their content comes from those runs. The
folder also holds the figures that `scripts/build_figures.py` draws from them, the coverage
report of the test suite under `coverage/`, and `run-evidence.json`, in which `scripts/smoke.py`
records its last run.

| File | What it holds | Written by |
|---|---|---|
| `evaluation_episodes.csv` | one row per episode of the exported grid, with its seed | `scripts/evaluate_and_export.py` |
| `evaluation_summary.json` | the metrics of that collection, and the spread across the six grids | `scripts/evaluate_and_export.py` |
| `evaluation_manifest.json` | the policy scored, the grids, the versions of Gymnasium, Stable-Baselines3, NumPy, PyTorch and Box2D, the git revision | `scripts/evaluate_and_export.py` |
| `seed_study.json` | the five trainings and the DQN at equal budget | `scripts/aggregate_study.py` |
| `hyperparameter_trials.csv` | one row per single-parameter trial | `scripts/aggregate_study.py` |
| `learning_curve_band.csv` | median and interquartile range of the five curves | `scripts/aggregate_study.py` |
| `checkpoint_comparison.csv` | the best and final checkpoints of two runs, scored on the exported grid | `scripts/compare_checkpoints.py` |
| `training_curves.csv` | the shipped run's own curve | `scripts/publish_run.py` |

`models/ppo/best.zip` is a Stable-Baselines3 checkpoint of about 148 KB. It holds weights that
this repository trained, on an architecture the library provides. The run directories under
`var/runs/` are not tracked. Each one holds the checkpoints of a run, its manifest and its
training curve.

## How far "produced locally" goes

Producing the data locally settles the question of redistribution, since no third party's data
is involved. Reproducibility is a separate question. The section on seeds above explains why a
rerun of `scripts/train_study.py` with the same seeds may differ from the committed weights, by
an amount that has not been measured. The spread between
runs, which `docs/protocol.md` defines and the README reports, is the variation to expect when
the configuration is trained with new seeds.

The manifest lets a reader check the published scores without any retraining.
`reports/evaluation_manifest.json` names the checkpoint, the six grids and the versions of
Gymnasium, Stable-Baselines3, NumPy, PyTorch and Box2D, and `scripts/evaluate_and_export.py`
scores that checkpoint again in about a minute.
