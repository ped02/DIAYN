import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EnvNavigation2D(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode: str = None):
        super().__init__()
        self.render_mode = render_mode

        # Define observation space (x, y)
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(2,), dtype=np.float32
        )

        # Define action space (Δx, Δy)
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(2,), dtype=np.float32
        )

        # Initialize state
        self.state = np.zeros(2, dtype=np.float32)
        self.reward_const = 1.0  # Constant reward
        self.max_steps = 100  # To prevent infinite episodes
        self.current_step = 0  # Track steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(2, dtype=np.float32)
        self.current_step = 0  # Reset step counter
        return self.state, {}  # (observation, info)

    def step(self, action):
        # Clip action and update state
        action = np.clip(
            action, self.action_space.low, self.action_space.high
        ).squeeze()
        self.state = np.clip(
            self.state + action,
            self.observation_space.low,
            self.observation_space.high,
        )

        # Define reward (optional customization)
        reward = self.reward_const  # -np.linalg.norm(self.state)  # Example: Negative distance from origin

        self.current_step += 1
        terminated = False  # np.linalg.norm(self.state) < 0.1  # Example: Task complete when close to (0,0)
        truncated = (
            self.current_step >= self.max_steps
        )  # Episode ends after max steps
        
        return (
            self.state,
            reward,
            terminated,
            truncated,
            {},
        )  # Gymnasium step return format

    def render(self):
        if self.render_mode == 'human':
            print(f'Current State: {self.state}')
        elif self.render_mode == 'rgb_array':
            pass  # Could return a visual representation as an array

    def close(self):
        pass
