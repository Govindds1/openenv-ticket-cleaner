from typing import Optional

from openenv.core.env_server.types import Action as OpenEnvAction
from openenv.core.env_server.types import Observation as OpenEnvObservation
from openenv.core.env_server.types import State as OpenEnvState
from pydantic import Field

class Observation(OpenEnvObservation):
    """What the agent sees."""

    data_preview: str = Field(..., description="Preview of the ticket dataset")
    current_task: str = Field(..., description="Current requested cleaning task")
    message: str = Field(..., description="Human-readable task instruction")

class Action(OpenEnvAction):
    """What the agent can do."""

    command: str = Field(..., description="Command to execute")
    column: Optional[str] = Field(default=None, description="Optional target column")
    value: Optional[str] = Field(default=None, description="Optional value parameter")

class State(OpenEnvState):
    """Internal environment state (separate from Observation)."""

    current_task_idx: int = Field(default=0, ge=0)