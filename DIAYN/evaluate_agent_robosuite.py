from typing import Optional

import numpy as np
import torch
import yaml

import gymnasium as gym

from DIAYN import AgentBase
from DIAYN.utils import pad_to_dim_2


import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

import robosuite as suite
from robosuite.wrappers import GymWrapper

class CustomGymWrapper(gym.ObservationWrapper):
    def __init__(self, robosuite_env, config):
        # Wrap robosuite with GymWrapper inside
        gym_env = GymWrapper(robosuite_env)
        super().__init__(gym_env)

        # Figure out which state variables we care about
        self.use_eef_state = config['observations']['use_eef_state']

        # Extract dimensions of observation space
        obs_dict = robosuite_env._get_observations()

        if self.use_eef_state:
            observation_raw = np.concatenate([
                obs_dict['robot0_eef_pos'],
                obs_dict['robot0_eef_quat']
            ])
        else:
            observation_raw = np.concatenate([
                obs_dict['robot0_proprio-state'],
                obs_dict['object-state']
            ])

        # Add new dimensions to observation spaceZ
        new_dim = observation_raw.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_dim,),
            dtype=np.float32
        )
    
    def observation(self, obs):
        # obs is the flat vector from GymWrapper
        # Add x-position (or whatever custom feature you want)
        obs_dict = 	self.env.unwrapped._get_observations()

        if (self.use_eef_state):
            # print("using end effector state")
            # print("Using end effector state")
            # print("eef pos: " + str(obs_dict['robot0_eef_pos']))
            # print("eef quat: " + str(obs_dict['robot0_eef_quat']))
            observation_raw = np.concatenate([obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat']])
        else:
            observation_raw = np.concatenate([obs_dict['robot0_proprio-state'], obs_dict['object-state']])

        # print("obs dict from custom gym wrapper: " + str(obs_dict))
        # print("observation_raw: " + str(observation_raw))
        return observation_raw

def make_env(env_name, robots):
    def _thunk():
        print("Creating new environment")

        controller_config = load_composite_controller_config(
                controller=None,
                robot="Panda",
        )

        robosuite_config = {
            "env_name": env_name,
            "robots": robots,
            "controller_configs": controller_config,
        }

        robosuite_env = suite.make(
            **robosuite_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=False,
        )

        with open("./config/ur5e_config.yaml", "r") as file:
            config = yaml.safe_load(file)

        env = CustomGymWrapper(robosuite_env, config)

        # Ensure metadata exists and is a dict before modifying
        if env.metadata is None:
            env.metadata = {}
        env.metadata["render_modes"] = []
        env.metadata["autoreset"] = False

        return env
    return _thunk

def evaluate_agent_robosuite(
    environment_name: str,
    robots: str,
    max_episode_steps: int,
    eval_episodes: int,
    agent: AgentBase,
    device,
    skill_index: Optional[int] = None,
    num_skills: Optional[int] = None,
):
    """Evaluate Agent Interaction. Run until episode ends or max_episode_step is reached.

    Args:
        environment_name (str): name of gym environment
        max_episode_steps (int): max number of steps in episode
        eval_episodes (int): Number of episodes to average
        agent (AgentBase): Agent to visualize
        device (str or pytorch device): Device of agent
        output_folder (str): Path to directory of output
        output_name_prefix (str): Name prefix of output file
    """

    skill_vector = None
    if skill_index is not None:
        skill_vector = torch.nn.functional.one_hot(
            torch.tensor(skill_index, device=device), num_classes=num_skills
        ).unsqueeze(0)

    def augment_state_skill(observation_raw):
        return torch.cat(
            [
                pad_to_dim_2(torch.Tensor(observation_raw).to(device)),
                skill_vector,
            ],
            dim=-1,
        )

    env = make_env(environment_name, robots)()

    episode_total_return = []
    episode_length = []

    def process_pure_state(observation_raw):
        return pad_to_dim_2(torch.Tensor(observation_raw).to(device))

    process_observation = process_pure_state
    if skill_index is not None:
        process_observation = augment_state_skill

    for episode in range(eval_episodes):
        observations_raw, _ = env.reset()
        observation = process_observation(observations_raw)

        episode_return = 0.0

        for t in range(max_episode_steps):
            with torch.no_grad():
                actions = agent.get_action(observation).squeeze(0)
            observations_raw, rewards, done, _, _ = env.step(
                actions.cpu().numpy()
            )
            observation = process_observation(observations_raw)
            episode_return += rewards
            if done:
                break

        episode_total_return.append(episode_return)
        episode_length.append(t)

    env.close()

    result_dict = {
        'total_return': np.array(episode_total_return),
        'episode_length': np.array(episode_length),
    }

    return result_dict
