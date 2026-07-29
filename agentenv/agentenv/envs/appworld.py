"""AppWorld env client for AgentGym. Talks to the agentenv-appworld FastAPI server.

Code-as-action under a ReAct wrapper: each turn the model emits Thought + Action,
where Action is a Python code block calling AppWorld APIs. The client extracts the
code and forwards it; the server's AppWorld.execute(code) returns printed output as
the next observation.
"""

from typing import Any

import requests

from agentenv.controller import (
    BaseAdapter,
    BaseEnvClient,
    BaseTask,
    extract_python_code_blocks,
)
from agentenv.controller.types import (
    ActionFormat,
    ConversationMessage,
    StepOutput,
)

_APPWORLD_SYSTEM = """You are an autonomous agent that completes tasks inside AppWorld by writing Python code that calls app APIs. There are 9 apps: api_docs, supervisor, amazon, phone, spotify, venmo, gmail, simba, todoist.

EVERY turn you MUST reply with exactly a Thought and an Action, where the Action is a single Python code block. Format:

Thought:
<your reasoning about what to do next>

Action:
```python
<python code calling the APIs>
```

How the environment works:
- You only SEE what you print(). Wrap any value you want to observe in print().
- The task and the supervisor's credentials come from the `supervisor` app:
    apis.supervisor.show_active_task()  -> the task instruction to accomplish
    apis.supervisor.show_account_passwords()/.show_profile()/.show_addresses()/.show_payment_cards()  -> info/credentials you may need
- You do NOT know any app's API signatures up front. Discover them on demand:
    apis.api_docs.show_app_descriptions()
    apis.api_docs.show_api_descriptions(app_name="spotify")
    apis.api_docs.show_api_doc(app_name="spotify", api_name="login")
  Always look up an API's doc before calling it, so you pass the right parameters.
- Most apps require logging in first with the supervisor's credentials (e.g.
  apis.spotify.login(username=..., password=...)) then passing the returned
  access_token to subsequent calls.

Finishing the task (this determines your score):
- When done you MUST call apis.supervisor.complete_task(...). Look up its exact
  parameters with api_docs first. For question-answering tasks pass your final
  answer to it; for action tasks call it once the required changes are made.
- Reward is 1 only if the task's hidden state-based tests all pass, else 0.

Work in small steps: inspect with print(), verify intermediate results, then act."""


class AppWorldAdapter(BaseAdapter):
    conversation_start_dict = {
        ActionFormat.REACT: (
            ConversationMessage({"from": "human", "loss": None, "value": _APPWORLD_SYSTEM}),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
    }


class AppWorldEnvClient(BaseEnvClient):
    adapter_cls = AppWorldAdapter

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 600, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise requests.RequestException(f"Failed to create environment: {ok}")
        self.env_id = ok.json()
        self.conversation_start = self.adapter_cls.conversation_start_dict[
            self.action_format
        ]

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict) -> Any:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}", json=data, timeout=self.timeout
        )
        assert res.status_code == 200, (res.status_code, res.text[:300])
        return res.json()

    def _get(self, path: str) -> Any:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}", timeout=self.timeout
        )
        assert res.status_code == 200, (res.status_code, res.text[:300])
        return res.json()

    def observe(self) -> str:
        return self._get("observation")

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        try:
            code = AppWorldAdapter.action_parser(action, self.action_format)
            code = extract_python_code_blocks(code)
        except Exception as e:
            print("appworld action parse error:", e)
            return StepOutput(state="Invalid Action.", reward=0.0, done=False)
        response = self._post("step", {"action": code})
        return StepOutput(
            state=response["state"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int) -> str:
        return self._post("reset", {"session_id": idx})


class AppWorldTask(BaseTask):
    env_client_cls = AppWorldEnvClient
    env_name = "AppWorld"

    def __init__(self, client_args: dict, n_clients: int = 1, *args, **kwargs) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
