# Agent Environments - AppWorld

## Setup

Requires Python 3.11+.

``` sh
conda create --name agentenv-appworld python=3.11
conda activate agentenv-appworld
pip install -e .
appworld install
appworld download data
```

## Launch

``` sh
appworld-env --host 0.0.0.0 --port 36001
```

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `APPWORLD_SPLIT` | `train` | dataset split used for the `session_id` -> `task_id` mapping |
| `APPWORLD_MAX_INTERACTIONS` | `50` | max `execute()` calls per episode |

## API

Actions are **Python code strings** (code-as-action). `AppWorld.execute(code)` returns
stdout or an error traceback as the next observation. Task success is a state-based
unit test via `AppWorld.evaluate()`; `done` comes from `AppWorld.task_completed()`.

| Route | Method | Payload / Query |
|---|---|---|
| `/` | GET | – |
| `/list_envs` | GET | – |
| `/create` | POST | – (returns `env_idx`) |
| `/reset` | POST | `{"env_idx": int, "session_id": int \| null}` |
| `/step` | POST | `{"env_idx": int, "action": "<python code>"}` |
| `/observation` | GET | `?env_idx=int` |
| `/instruction_text` | GET | `?env_idx=int` |
