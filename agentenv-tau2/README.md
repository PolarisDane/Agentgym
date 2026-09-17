# agentenv-tau2

AgentGym environment server for [tau2-bench](https://github.com/sierra-research/tau2-bench)
(τ²/τ³-bench). It wraps tau2's Gymnasium env (`tau2.gym.gym_agent.AgentGymEnv`) in the
AgentGym env-server HTTP contract so AgentGym-RL / verl can train against it unchanged.

Runs as its own process because tau2 requires Python >= 3.12 while the training
environment is on Python 3.10.

## Install

```bash
conda create -y -p /path/to/agentenv-tau2 python=3.12
/path/to/agentenv-tau2/bin/pip install -e /path/to/tau2-bench
/path/to/agentenv-tau2/bin/pip install -e /path/to/AgentGym/agentenv-tau2
```

## Run

```bash
TAU2_DOMAIN=retail \
TAU2_TASK_SPLIT=train \
TAU2_USER_LLM=openai/user-sim \
TAU2_USER_API_BASE=http://127.0.0.1:38001/v1 \
tau2-env --host 127.0.0.1 --port 36201
```

## Endpoints

| Method | Path | Body / Query | Returns |
|---|---|---|---|
| GET  | `/`              | –                      | `"ok"` |
| GET  | `/system_prompt` | –                      | domain policy + tool list + response format |
| POST | `/create`        | –                      | `env_idx` |
| POST | `/reset`         | `{env_idx, session_id}`| opening observation |
| POST | `/step`          | `{env_idx, action}`    | `{state, reward, done, info}` |
| GET  | `/observation`   | `?env_idx=`            | last observation |
| POST | `/close`         | `{env_idx}`            | `"ok"` |

`session_id` is an integer; it is mapped to a tau2 task id by
`task_ids[session_id % len(task_ids)]` over the configured domain/split.

## Configuration

See the module docstring in `agentenv_tau2/environment.py` for the full list of
`TAU2_*` environment variables.

Two defaults worth knowing:

- **`TAU2_REWARD_BASIS=env`** — tau2's own `AgentGymEnv` hardcodes `EvaluationType.ALL`,
  which invokes an NL-assertion judge LLM for any task whose `reward_basis` includes
  `NL_ASSERTION` (112 of retail's 114 tasks). `env` scores from DB/env state only:
  deterministic, free, and no judge noise in the RL advantage. Use `all` to reproduce
  the official leaderboard metric at eval time.
- **`all_messages_as_observation=False`** — makes `step()` return only the messages
  after the last assistant message. verl's `RolloutHandler` accumulates the
  conversation itself, so returning full history would grow the context quadratically.
