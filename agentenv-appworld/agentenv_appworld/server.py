"""
FastAPI server for the AppWorld agent environment. Mirrors the webshop server's
contract so the AgentGym client/controller can talk to it unchanged:
  POST /create              -> env_idx (int)
  POST /reset  {env_idx, session_id}  -> instruction (str)
  POST /step   {env_idx, action}      -> {state, reward, done, info}
  POST /close  {env_idx}              -> None
  GET  /observation?env_idx=          -> str
  GET  /instruction_text?env_idx=     -> str
"""

import logging
import time
from typing import List

from fastapi import FastAPI, Request

from .environment import appworld_env_server
from .model import CloseQuery, ResetQuery, StepQuery, StepResponse
from .utils import debug_flg

app = FastAPI(debug=debug_flg)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


@app.middleware("http")
async def log_request_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(
        f"{request.client.host} - {request.method} {request.url.path} - "
        f"{response.status_code} - {process_time:.2f}s"
    )
    return response


@app.get("/", response_model=str)
async def generate_ok():
    return "ok"


@app.get("/list_envs", response_model=List[int])
async def list_envs():
    return list(appworld_env_server.env.keys())


@app.post("/create", response_model=int)
async def create():
    return appworld_env_server.create()


# NOTE: /reset and /step are async so they run on uvicorn's main-thread event loop.
# AppWorld.execute() installs a SIGALRM timeout handler, and signal.signal() only works
# in the main thread -- a sync `def` endpoint would run in FastAPI's threadpool and crash
# with "signal only works in main thread". The AppWorld calls are blocking/CPU-bound; env
# interactions are serial per env, so blocking the loop briefly is acceptable here.
@app.post("/reset", response_model=str)
async def reset(reset_query: ResetQuery):
    return appworld_env_server.reset(reset_query.env_idx, reset_query.session_id)


@app.post("/step", response_model=StepResponse)
async def step(step_query: StepQuery):
    state, reward, done, info = appworld_env_server.step(
        step_query.env_idx, step_query.action
    )
    return StepResponse(state=state, reward=reward, done=done, info=info)


# AppWorld.close() 清理进程级全局状态(clear_local_dbs_cache / id_to_time_freezer /
# ApiCollection)。不调用它，实例会逐轮累积并互相干扰，_save_state 会报
# FileNotFoundError: .../dbs/model_hashes.json。verl 每轮 rollout 结束会调
# client.close()，必须有对应端点。
@app.post("/close")
async def close(close_query: CloseQuery):
    appworld_env_server.close(close_query.env_idx)
    return None


@app.get("/observation", response_model=str)
def observation(env_idx: int):
    return appworld_env_server.observation(env_idx)


@app.get("/instruction_text", response_model=str)
def instruction_text(env_idx: int):
    return appworld_env_server.get_instruction_text(env_idx)
