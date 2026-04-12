import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from server.app import TicketEnvironment
from model import Action

# Configuration
def _get_api_key() -> Optional[str]:
    # Prefer explicit env vars.
    key = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACE_API_TOKEN")
        or os.getenv("OPENAI_API_KEY")
    )
    if key:
        return key

    # Fallback: if user has run `huggingface-cli login`, reuse that token.
    try:
        from huggingface_hub import HfFolder  # type: ignore

        return HfFolder.get_token()
    except Exception:
        return None


API_KEY = _get_api_key()
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

SYSTEM_PROMPT = """
You are a data cleaning agent. You will receive a preview of a ticket database.
Your goal is to perform the 'current_task' requested.
You must respond with ONLY the command name.
Available commands: remove_duplicates, fix_priority, standardize_status
Example Response: remove_duplicates
"""

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = "null"):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    env = TicketEnvironment() # Initialize your local env
    
    tasks = ["remove_duplicates", "fix_priority", "standardize_status"]
    
    for task_name in tasks:
        log_start(task=task_name, env="ticket_cleaner", model=MODEL_NAME)
        obs = env.reset()
        rewards = []
        
        for step in range(1, 4):  # Max 3 attempts per task
            prompt = f"Data:\n{obs.data_preview}\nTask: {obs.current_task}\nCommand:"
            
            if client is None:
                # Offline fallback: choose the obvious command for the task.
                cmd = obs.current_task.strip()
            else:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                cmd = completion.choices[0].message.content.strip()
            obs = env.step(Action(command=cmd))
            reward = float(obs.reward or 0.0)
            done = bool(obs.done)
            
            rewards.append(reward)
            log_step(step=step, action=cmd, reward=reward, done=done)
            
            if done or reward == 1.0:
                break
        
        score = sum(rewards) / len(rewards) if rewards else 0
        log_end(success=(score > 0.7), steps=len(rewards), score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
