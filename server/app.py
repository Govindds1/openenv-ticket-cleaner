import pandas as pd
import uvicorn
from typing import Optional, List
from fastapi import FastAPI, Request
from pydantic import BaseModel

# --- MODELS (Embedded to prevent import errors) ---
class State(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    current_task_idx: int = 0

class Action(BaseModel):
    command: str

class Observation(BaseModel):
    done: bool
    reward: Optional[float]
    data_preview: str
    current_task: str
    message: str

# --- ENVIRONMENT LOGIC ---
class TicketEnvironment:
    def __init__(self):
        self._state = State()
        self.reset()

    def reset(self, episode_id: Optional[str] = None) -> Observation:
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
        return self._observation(reward=0.0, done=False)

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
        # Advance task if reward is achieved
        if reward >= 1.0 and self._state.current_task_idx < len(self.tasks) - 1:
            self._state.current_task_idx += 1
            
        done = (self._state.current_task_idx == len(self.tasks) - 1 and reward >= 1.0) or self._state.step_count > 10
        return self._observation(reward=reward, done=done)

    def get_reward(self) -> float:
        if self._state.current_task_idx == 0:
            return 1.0 if not self.df.duplicated().any() else 0.0
        if self._state.current_task_idx == 1:
            return 1.0 if self.df["priority"].notnull().all() else 0.0
        return 1.0 if self.df["status"].str.islower().all() else 0.0

# --- FASTAPI SERVER ---
app = FastAPI()
env = TicketEnvironment()

@app.get("/")
async def health():
    return {"status": "ok", "env": "TicketCleaner"}

@app.post("/reset")
async def reset_endpoint():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
async def step_endpoint(action: Action):
    obs = env.step(action)
    return obs.model_dump()

@app.get("/state")
async def state_endpoint():
    return env._state.model_dump()
def main():
    """Entry point for the OpenEnv validator"""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
