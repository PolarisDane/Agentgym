"""
AppWorldEnvServer -- wraps the AppWorld engine into the AgentGym env-server contract
(create / reset / step / observation), mirroring WebshopEnvServer.

Each env slot holds one live AppWorld instance. The agent's action is a Python code
string (code-as-action); AppWorld.execute(code) returns a text string (stdout or an
error traceback) which is the next observation. Task success is a state-based unit
test via AppWorld.evaluate(); done is AppWorld.task_completed().

Config via env vars:
  APPWORLD_SPLIT           dataset split for task-id mapping (default: train)
  APPWORLD_MAX_INTERACTIONS max execute() calls per episode (default: 50)
"""

import os
from typing import Optional, Tuple

from appworld import AppWorld, load_task_ids


class AppWorldEnvServer:
    def __init__(self) -> None:
        self._max_id = 0
        self.env = {}            # env_idx -> AppWorld | None
        self.step_count = {}     # env_idx -> interactions used this episode
        self._reset_count = 0
        self.split = os.environ.get("APPWORLD_SPLIT", "train")
        self.max_interactions = int(os.environ.get("APPWORLD_MAX_INTERACTIONS", "50"))
        # ordered list of task ids for integer session_id -> task_id mapping
        self._task_ids = load_task_ids(self.split)
        if not self._task_ids:
            raise RuntimeError(f"AppWorld: no task ids for split '{self.split}'")

    # ---- lifecycle ----------------------------------------------------------
    def create(self) -> int:
        env_idx = self._max_id
        self._max_id += 1
        self.env[env_idx] = None  # not yet reset to a task
        print(f"-------AppWorld env slot {env_idx} created--------")
        return env_idx

    def _task_id_for(self, session_id: Optional[int]) -> str:
        sid = 0 if session_id is None else int(session_id)
        return self._task_ids[sid % len(self._task_ids)]

    def _close_slot(self, env_idx: int) -> None:
        world = self.env.get(env_idx)
        if world is not None:
            try:
                world.close()
            except Exception:
                pass
            self.env[env_idx] = None

    def reset(self, env_idx: int, session_id: Optional[int]) -> str:
        """(Re)bind this slot to a task; return the task instruction as first obs."""
        self._close_slot(env_idx)
        task_id = self._task_id_for(session_id)
        self._reset_count += 1
        world = AppWorld(
            task_id=task_id,
            experiment_name=f"agentenv_{env_idx}_{self._reset_count}",
            max_interactions=self.max_interactions,
        )
        self.env[env_idx] = world
        self.step_count[env_idx] = 0
        return world.task.instruction

    # ---- interaction --------------------------------------------------------
    def step(self, env_idx: int, action: str) -> Tuple[str, float, bool, None]:
        world = self.env.get(env_idx)
        if world is None:
            return "Environment not reset. Call /reset first.", 0.0, False, None
        # action is a python code string (code-as-action)
        state = world.execute(action)
        self.step_count[env_idx] = self.step_count.get(env_idx, 0) + 1
        # done on task completion OR when the interaction budget is exhausted
        done = bool(world.task_completed()) or (
            self.step_count[env_idx] >= self.max_interactions
        )
        reward = 0.0
        if done:
            # ORM-style sparse outcome reward: 1.0 iff all state-based tests pass, else 0.
            try:
                result = world.evaluate().to_dict()
                reward = 1.0 if result.get("success") else 0.0
            except Exception as e:
                print(f"[appworld] evaluate() failed on env {env_idx}: {e}")
                reward = 0.0
        return str(state), reward, done, None

    def observation(self, env_idx: int) -> str:
        world = self.env.get(env_idx)
        if world is None:
            return ""
        # first-turn observation == the task instruction; kept simple for the skeleton
        return world.task.instruction

    def get_instruction_text(self, env_idx: int) -> str:
        world = self.env.get(env_idx)
        return world.task.instruction if world is not None else ""

    def close(self, env_idx: int) -> None:
        self._close_slot(env_idx)
        print(f"-------AppWorld env slot {env_idx} closed--------")

    def __del__(self):
        for idx in list(self.env.keys()):
            self._close_slot(idx)


appworld_env_server = AppWorldEnvServer()
