# 🎫 TicketCleaner-OpenEnv
**A Real-World Data Operations Environment for AI Agents**

[![OpenEnv Spec](https://img.shields.io/badge/Spec-OpenEnv--v1-blue)](https://github.com/openenv/spec)
[![Deploy to HF](https://img.shields.io/badge/%F0%9F%A4%97-Deploy%20to%20Spaces-yellow)](https://huggingface.co/spaces/Govindds/ticket-cleaner-env)

## 📖 Overview
TicketCleaner is a high-fidelity simulation of a **Customer Support Data Operations** task. AI agents are tasked with cleaning, imputing, and standardizing a messy ticket database. Unlike toy environments, this reflects the actual workflows of data engineers and support leads.

## 🛠️ Environment Specification
Built on the **OpenEnv** framework, the environment uses a standard `step()` / `reset()` / `state()` API with Pydantic-validated models.

### 🔭 Observation Space
The agent receives a `data_preview` (a string representation of the current DataFrame) and the `current_task` description.

### 🎮 Action Space
The agent interacts with the environment using the following commands:
| Command | Effect |
| :--- | :--- |
| `remove_duplicates` | Removes identical row entries. |
| `fix_priority` | Fills `NaN` priority values with a default "Medium" tag. |
| `standardize_status` | Normalizes all status entries to lowercase. |

## 🎯 Task Suite & Grading
The environment features three tasks with programmatic reward signals ($0.0$ to $1.0$):

1. **Task 1 (Easy
