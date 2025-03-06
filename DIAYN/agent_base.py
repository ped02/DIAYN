from typing import Protocol, Any

class AgentBase(Protocol):
    def get_action(self, state: Any, noise: bool) -> Any:
        ...

    def pre_episode(self):
        ...