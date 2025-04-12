import os
from typing import Optional, Union
import yaml
import numpy as np
import torch
import gymnasium as gym

from DIAYN import DIAYNAgent, evaluate_agent, evaluate_agent_robosuite, visualize, visualize_robosuite

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

from gymnasium.vector import SyncVectorEnv
import robosuite as suite
from robosuite.wrappers import GymWrapper

from DIAYN.utils import (
    replay_post_processor,
    pad_to_dim_2,
    plot_to_image,
    image_numpy_to_torch,
)

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

def make_env():
    def _thunk():
        print("Creating new environment")

        controller_config = load_composite_controller_config(
                controller=None,
                robot="Panda",
        )

        robosuite_config = {
            "env_name": "Lift",
            "robots": "Panda",
            "controller_configs": controller_config,
        }

        robosuite_env = suite.make(
            **robosuite_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            render_camera="birdview",
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

def main(
    environment_name: str,
    robots: str,
    steps_per_episode: int,
    num_skills: int,
    visualize_skill: Union[int, None],
    video_output_folder: str,
    video_file_prefix: str = 'rl_video',
    model_load_path: Optional[str] = None,
    evaluate_episodes: int = 10,
    config: Optional[dict] = None,
):
    device = torch.device('cuda')

    # Setup logging
    log_writer = None

    # Setup env variables

    # TODO: Just get dims directly instead of dealing with this each time
    envs = SyncVectorEnv([make_env() for _ in range(num_envs)])
    
    observation_dims = envs.observation_space.shape[1]
    action_dims = envs.action_space.shape[1]

    action_low = envs.action_space.low[0]
    action_high = envs.action_space.high[0]

    # Setup networks
    def get_q_network(observation_dim, action_dim):
        q_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim + action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

        return q_network

    def get_policy_network(observation_dim, action_dim):
        policy_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2 * action_dim),
        )
        return policy_network

    def get_discriminiator_network(observation_dim, skill_dim):
        # Output logits
        discriminiator_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, skill_dim),
        )

        return discriminiator_network

    q_optimizer_kwargs = {'lr': 1e-3}
    discriminator_optimizer_kwargs = {'lr': 4e-4}
    policy_optimizer_kwargs = {'lr': 1e-4}

    # Setup agent
    diayn_agent = DIAYNAgent(
        skill_dim=num_skills,
        discriminator_class=get_discriminiator_network,
        discriminator_optimizer_kwargs=discriminator_optimizer_kwargs,
        state_dim=observation_dims,
        action_dim=action_dims,
        policy_class=get_policy_network,
        q_network_class=get_q_network,
        alpha=0.1,
        action_low=action_low,
        action_high=action_high,
        policy_optimizer_kwargs=policy_optimizer_kwargs,
        q_optimizer_kwargs=q_optimizer_kwargs,
        device=device,
        log_writer=log_writer,
    )

    if model_load_path is not None:
        diayn_agent.load_checkpoint(model_load_path)

    # Skill
    if visualize_skill is None:
        skills_mean_step_reward = []
        skills_return = []
        for z in range(num_skills):
            visualize_robosuite(
                environment_name,
                robots,
                steps_per_episode,
                diayn_agent,
                device,
                skill_index=z,
                num_skills=num_skills,
                output_folder=video_output_folder,
                output_name_prefix=f'{video_file_prefix}_skill{z}',
                config=config,
            )

            result_dict = evaluate_agent_robosuite(
                environment_name,
                robots,
                steps_per_episode,
                evaluate_episodes,
                diayn_agent,
                device,
                skill_index=z,
                num_skills=num_skills,
            )
            skills_return.append(np.mean(result_dict['total_return']))
            skills_mean_step_reward.append(
                np.nanmean(
                    result_dict['total_return'] / result_dict['episode_length']
                )
            )
            print(
                f'[Skill {z}] Mean Step Reward: {skills_mean_step_reward[-1]} Mean Total Return: {skills_return[-1]}'
            )

        with open(
            os.path.join(video_output_folder, 'skills_order_step_reward'), 'w'
        ) as f:
            f.write(
                '\n'.join(
                    [
                        f'{skill}: {total_return}'
                        for skill, total_return in sorted(
                            enumerate(skills_mean_step_reward),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    ]
                )
            )

        with open(
            os.path.join(video_output_folder, 'skills_order_total_return'), 'w'
        ) as f:
            f.write(
                '\n'.join(
                    [
                        f'{skill}: {total_return}'
                        for skill, total_return in sorted(
                            enumerate(skills_return),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    ]
                )
            )

    else:
        visualize_robosuite(
            environment_name,
            robots,
            steps_per_episode,
            diayn_agent,
            device,
            skill_index=visualize_skill,
            num_skills=num_skills,
            output_folder=video_output_folder,
            output_name_prefix=f'{video_file_prefix}_skill{visualize_skill}',
        )
        result_dict = evaluate_agent_robosuite(
            environment_name,
            robots,
            steps_per_episode,
            evaluate_episodes,
            diayn_agent,
            device,
            skill_index=visualize_skill,
            num_skills=num_skills,
        )
        print(
            f'[Skill {visualize_skill}] Mean Step Reward: {np.nanmean(result_dict["total_return"]/result_dict["episode_length"])} Mean Total Return: {np.mean(result_dict["total_return"])}'
        )

def checkpoint_check(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file '{filepath}' not found")

    checkpoint = torch.load(filepath, map_location='cuda')
    print("Checkpoint contents:", checkpoint)  # Add this line to debug
    # self.set_state_dict(checkpoint)

    # logger.info(f'Checkpoint loaded from {filepath}')

if __name__ == '__main__':

    # Read config file for all settings
    with open("./config/ur5e_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    environment_name = config['params']['environment_name']
    robots = config['params']['robots']
    num_envs = config['params']['num_envs']
    num_steps = config['evaluation_params']['num_steps']
    num_skills = config['params']['num_skills']

    # Folder paths
    model_load_path = config['file_params']['model_load_path']
    video_output_folder = config['file_params']['video_output_folder']
    video_prefix_path = config['file_params']['video_prefix_path']

    # If we want a specific skill to be visualized
    visualize_skill = None

    if video_output_folder is not None:
        os.makedirs(video_output_folder, exist_ok=True)

    main(
        environment_name,
        robots,
        num_steps,
        num_skills,
        visualize_skill,
        video_output_folder,
        video_prefix_path,
        model_load_path=model_load_path,
        config=config
    )
