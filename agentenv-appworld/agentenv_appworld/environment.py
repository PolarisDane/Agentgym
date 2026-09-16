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

import json
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
        """释放 slot：只丢引用，**不要**调 AppWorld.close()。

        AppWorld.__init__ -> initialize -> close_all() 自己就会清进程级全局状态
        (clear_local_dbs_cache / id_to_time_freezer / ApiCollection)。若我们再显式
        close 一次，freezegun 的时间冻结栈会被弹空，下一次构造 AppWorld 直接抛
        IndexError: pop from empty list —— 实测顺序跑 8 个 episode 全部 reset 500。
        丢引用即可让对象被 GC，全局状态由下一次 __init__ 的 close_all() 接管。
        """
        if self.env.get(env_idx) is not None:
            self.env[env_idx] = None

    def reset(self, env_idx: int, session_id: Optional[int]) -> str:
        """(Re)bind this slot to a task; return the task instruction as first obs."""
        self._close_slot(env_idx)
        task_id = self._task_id_for(session_id)
        self._reset_count += 1
        world = AppWorld(
            task_id=task_id,
            # 2026-09-01: 加 PID。env_idx/_reset_count 都是每进程各自从 0 开始的计数器，
            # 而 128 个 server 进程写同一个 experiments/outputs/ 目录 -> 目录名跨进程
            # 碰撞，_save_state 的 save_local_dbs(delete_if_exists=True) 会删掉别的
            # 进程正在用的目录，受害进程下一步报 FileNotFoundError: .../model_hashes.json。
            # 实测: 128 进程 x 36 步 = 4608 个 episode 只落了 487 个目录，约 90% 撞名。
            experiment_name=f"agentenv_{os.getpid()}_{env_idx}_{self._reset_count}",
            max_interactions=self.max_interactions,
        )
        self.env[env_idx] = world
        self.step_count[env_idx] = 0
        # 随任务指令一并回传本 episode 的真实 supervisor 身份。官方 ReAct prompt 的
        # few-shot 演示里用 {{ main_user.email }} 硬编码登录，必须填当前 episode 的
        # 真实邮箱；填固定假值会被模型照抄，导致 401。
        sup = getattr(world.task, "supervisor", None)
        if sup is not None:
            info = {k: getattr(sup, k, "") for k in
                    ("first_name", "last_name", "email", "phone_number")}
            return json.dumps({"instruction": world.task.instruction,
                               "supervisor": info})
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
        return self._cap_observation(str(state)), reward, done, None

    # 观测长度上限。模型可以写出 print(巨大列表) 这类动作，一次观测就能吐出十几万
    # token：实测有轨迹上下文冲到 186486 token，vLLM 的 block manager 装不下 ->
    # 拒绝调度并返回空结果 -> verl decode 时抛
    # TypeError: argument 'ids': 'float' object cannot be interpreted as an integer。
    # LOOP 论文的做法是 "API responses exceeding 3K tokens are truncated, with a brief
    # note indicating the truncation"，这里对齐：按字符估算 ~3K token = 12000 字符，
    # 保留头尾（尾部常含真正的错误信息），中间用一行说明替换。
    _OBS_MAX_CHARS = int(os.environ.get("APPWORLD_OBS_MAX_CHARS", "12000"))

    def _cap_observation(self, text: str) -> str:
        n = len(text)
        if n <= self._OBS_MAX_CHARS:
            return text
        head = self._OBS_MAX_CHARS * 2 // 3
        tail = self._OBS_MAX_CHARS - head
        return (
            text[:head]
            + f"\n\n... [输出过长，已截断 {n - self._OBS_MAX_CHARS} 个字符。"
              f"请缩小查询范围或分页获取，不要一次 print 大量数据] ...\n\n"
            + text[-tail:]
        )

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
