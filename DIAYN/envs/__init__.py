import gymnasium as gym

from .env_navigation_2d import EnvNavigation2D

gym.register(
    id='Navigation2D-v0',
    entry_point=EnvNavigation2D,
)
