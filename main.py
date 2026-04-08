import pandas as pd
from typing import Optional

from openenv.core.env_server.http_server import create_app

from model import Action, Observation, State

class TicketEnvironment:
    def __init__(self):
        self._state = State()
        self.reset()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> Observation:
        # Create messy data: Duplicates, missing values, inconsistent casing
        self.df = pd.DataFrame([
            {"id": 1, "task": "Login issue", "priority": "High", "status": "open"},
            {"id": 1, "task": "Login issue", "priority": "High", "status": "open"}, # Duplicate
            {"id": 2, "task": "Broken link", "priority": None, "status": "OPEN"},   # Missing/Casing
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

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs) -> Observation:
        # Logic to handle the AI's commands
        if action.command == "remove_duplicates":
            self.df = self.df.drop_duplicates()
        elif action.command == "fix_priority":
            self.df["priority"] = self.df["priority"].fillna("Medium")
        elif action.command == "standardize_status":
            self.df["status"] = self.df["status"].str.lower()

        # Calculate Reward (0.0 to 1.0)
        reward = self.get_reward()
        
        # Move to next task if this one is done
        self._state.step_count += 1
        self._state.current_task_idx = min(self._state.current_task_idx + 1, len(self.tasks) - 1)
        
        done = self._state.current_task_idx == len(self.tasks) - 1 and reward == 1.0
        return self._observation(reward=reward, done=done)

    def get_reward(self) -> float:
        # Easy Grader: Check if duplicates are gone
        if self._state.current_task_idx == 0:
            return 1.0 if not self.df.duplicated().any() else 0.0
        # Medium Grader: Check if priority is filled
        if self._state.current_task_idx == 1:
            return 1.0 if self.df["priority"].notnull().all() else 0.5
        # Hard Grader: Check if all status are lowercase
        return 1.0 if self.df["status"].str.islower().all() else 0.0

    @property
    def state(self) -> State:
        return self._state


# This part connects it to the OpenEnv server
app = create_app(
    TicketEnvironment,
    Action,
    Observation,
    env_name="ticket_cleaner_env",
    max_concurrent_envs=1,
)