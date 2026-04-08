import pandas as pd
from typing import Optional
from fastapi import FastAPI, Request
from model import Action, Observation, State

# --- ENVIRONMENT LOGIC ---
class TicketEnvironment:
    def __init__(self):
        self._state = State()
        self.reset()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> Observation:
        self.df = pd.DataFrame([
            {"id": 1, "task": "Login issue", "priority": "High", "status": "open"},
            {"id": 1, "task": "Login issue", "priority": "High", "status": "open"},
            {"id": 2, "task": "Broken link", "priority": None, "status": "OPEN"},
            {"id": 3, "task": "Payment fail", "priority": "Low", "status": "closed"}
        ])
        self._state.episode_id = episode_id
        self._state.step_count = 0
        self._state.current_task_idx = 0
        self.tasks = ["remove_duplicates", "fix_priority", "standardize_status"]
        return self._observation(reward=None, done=False)

    def _observation(self, reward: Optional[float], done: bool) -> Observation:
        return Observation(
            done=done,
            reward=reward,
            data_preview=self.df.to_string(),
            current_task=self.tasks[self._state.current_task_idx],
            message=f"Please perform: {self.tasks[self._state.current_task_idx]}",
        )

    def step(self, action: Action) -> Observation:
        if action.command == "remove_duplicates":
            self.df = self.df.drop_duplicates()
        elif action.command == "fix_priority":
            self.df["priority"] = self.df["priority"].fillna("Medium")
        elif action.command == "standardize_status":
            self.df["status"] = self.df["status"].str.lower()
        
        reward = self.get_reward()
        self._state.step_count += 1
        self._state.current_task_idx = min(self._state.current_task_idx + 1, len(self.tasks) - 1)
        done = self._state.current_task_idx == len(self.tasks) - 1 and reward == 1.0
        return self._observation(reward=reward, done=done)

    def get_reward(self) -> float:
        if self._state.current_task_idx == 0:
            return 1.0 if not self.df.duplicated().any() else 0.0
        if self._state.current_task_idx == 1:
            return 1.0 if self.df["priority"].notnull().all() else 0.5
        return 1.0 if self.df["status"].str.islower().all() else 0.0

# --- FASTAPI WRAPPER (This fixes the 500/404 error) ---
app = FastAPI()
global_env = TicketEnvironment()

@app.get("/")
async def health():
    return {"status": "ok"}

@app.post("/reset")
async def reset_endpoint(request: Request):
    # This directly answers the 'openenv reset post' check
    obs = global_env.reset()
    return obs.dict()

@app.post("/step")
async def step_endpoint(action: Action):
    obs = global_env.step(action)
    return obs.dict()

@app.get("/state")
async def state_endpoint():
    return global_env._state.dict()
