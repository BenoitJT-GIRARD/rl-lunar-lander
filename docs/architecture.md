# How the policy is served, and where the boundaries are

This page describes how the service and the two Streamlit pages obtain the policy and the
results, and which components the repository leaves out. It assumes the README has been read.

## Components

```
        models/ppo/best.zip                         reports/*.csv
          │               │                               │
          ▼               ▼                               ▼
   ┌──────────────┐   ┌────────────────┐        ┌──────────────────┐
   │  cockpit     │──▶│  FastAPI       │        │   dashboard      │
   │  (Streamlit) │   │  rl_lander.api │        │   (Streamlit)    │
   └──────────────┘   └────────────────┘        └──────────────────┘
          │                    │                          │
          └────────────────────┴──────────────────────────┘
                               │
                  src/rl_lander/, the shared package
```

The service and the two pages import the same package, `src/rl_lander/`. The cockpit asks the
API for a trajectory. When its API toggle is off, or when the API does not answer or answers
with an error, the cockpit loads `models/ppo/best.zip` and plays the episode itself. The dashboard reads the published
exports under `reports/` and runs no policy.

## Why the policy is served by an API

**The checkpoint is loaded once.** Loading means unpacking a Stable-Baselines3 archive and
rebuilding its PyTorch network. The service does this at start-up, in its `lifespan` function,
and every request then uses the loaded policy. A page that played the policy itself would load
its own copy. The cockpit does so in its local mode, and keeps that copy in a
`st.cache_resource` cache.

**Two loops play episodes.** `LunarLanderAgent.play_episode`, in `rl_lander.agent`, plays the
episodes of the API, of the cockpit's local mode and of the video recorder. The published
evaluation uses a second loop, `run_episodes` in `training/evaluate.py`, which also records the
seed and the engine counts of each episode. Both loops used to read `landed` from the score,
and a single commit changed both to read the environment's terminal reward.

**The page can be replaced.** In API mode the cockpit knows the agent only through the JSON
contract of `/run`. Its local mode and its replay call `LunarLanderAgent` and `rl_lander.replay`,
which contain no Streamlit code, so a page written with another framework could call the same
functions unchanged.

**The API is tested without a browser.** `TestClient` sends requests to the application inside
the test process. The tests in `tests/integration/test_api.py` cover validation, status codes
and the case where no checkpoint is found, and they run in a few seconds.

## Why the cockpit renders the frames itself

`POST /run` returns the trajectory: the actions, the rewards, the total reward, the length, the
landing flag and the final state. It returns no images. A thousand RGB frames of 600×400 pixels
take 720 MB before any encoding, and the cockpit rebuilds them by replaying the actions in its
own copy of the environment.

The local replay shows the service's episode only if both follow the same trajectory, so the
cockpit checks this. `replay_matches`, in `rl_lander.replay`, compares the rewards of the replay
with the rewards the service returned, step by step, within an absolute tolerance of 10⁻⁶. When
they differ, the cockpit shows a red error message above the animation.

## Why the dashboard reads the published files

The dashboard shows the published evaluation. Its inputs are `reports/evaluation_episodes.csv`,
written by `scripts/evaluate_and_export.py`, and `reports/training_curves.csv`, copied there by
`scripts/publish_run.py`. The CSV and the metrics of the evaluation summary come from the same
call to `run_episodes`, and both files are tracked by git. The page therefore shows the hundred
episodes behind the README's rows for the shipped run and for grid 2024.

`rl_lander/artifacts.py` declares the columns of each export, outside the Streamlit module, and
the dashboard checks them before it draws its first panel. An earlier version looked up columns
while it drew the page, so an export from an older exporter failed with a `KeyError` partway
down. `tests/unit/test_artifacts_schema.py` covers the cases the check reports: a missing file,
an empty file, a header with no rows, and an export that lacks columns.

## Liveness and readiness

`/health` answers 200 while the process is serving. `/ready` answers 200 only when a policy is
loaded, and 503 otherwise. An orchestrator restarts a process whose liveness check fails, and
stops sending traffic to a process whose readiness check fails. In `rl_lander/api.py`, the
docstring of `ready` describes what happened when a single route answered both questions: a
load balancer kept sending requests to a service that returned 503 to each of them.

## Alternatives the repository leaves out

**A model registry.** The study has nine training runs and one shipped policy.
`scripts/publish_run.py` copies one run's `best.zip` and `manifest.json` to `models/ppo/`, so the
shipped checkpoint keeps the record of the run it came from. A registry such as MLflow would add
infrastructure around a choice that one command makes.

**A queue, workers or container orchestration.** One inference takes at most a few milliseconds
on the CPU, and uvicorn runs the service as a single process. The service is launched manually
when a demonstration or the cockpit needs it, and one process covers that use.

**A database.** The published exports are CSV and JSON files, small enough to read and to diff
in git. The choice follows from their size, and a run producing millions of rows would call for
a database.
