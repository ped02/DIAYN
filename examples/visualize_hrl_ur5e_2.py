import os
from typing import Optional, Union
import yaml

import numpy as np
import torch
from pathlib import Path

from DIAYN import DIAYNAgent, SAC, make_env, visualize_robosuite


from gymnasium.vector import SyncVectorEnv


def load_skill_policy(
    observation_dims,
    action_dims,
    action_low,
    action_high,
    device,
    model_load_path,
    log_writer,
):
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
            torch.nn.Linear(observation_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2 * action_dim),
        )
        return policy_network

    def get_discriminiator_network(observation_dim, skill_dim):
        # Output logits
        discriminiator_network = torch.nn.Sequential(
            torch.nn.LayerNorm(observation_dim),
            torch.nn.Linear(observation_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            torch.nn.Linear(128, skill_dim),
        )

        return discriminiator_network

    q_optimizer_kwargs = {'lr': 1e-3}
    discriminator_optimizer_kwargs = {'lr': 4e-4}
    policy_optimizer_kwargs = {'lr': 1e-4}

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

    diayn_agent.freeze()

    return diayn_agent


class HRLPolicy(torch.nn.Module):
    def __init__(
        self,
        observation_dim,
        skill_dim,
        sparsity_loss_factor: float = 1.0,
        **kwargs,
    ):
        super(HRLPolicy, self).__init__(**kwargs)

        self.skill_selector = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2 * skill_dim),
        )

        self.register_buffer(
            'sparsity_loss_factor', torch.tensor(sparsity_loss_factor)
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict, assign)

    def forward(self, states):
        return self.skill_selector(states)


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
    envs = SyncVectorEnv([make_env(config) for _ in range(num_envs)])
    observation_dims = envs.observation_space.shape[1]
    action_dims = envs.action_space.shape[1]

    action_low = envs.action_space.low[0]
    action_high = envs.action_space.high[0]

    # Load skills
    skill_policy = load_skill_policy(
        observation_dims,
        action_dims,
        action_low,
        action_high,
        device,
        skill_policy_path,
        log_writer,
    )

    def transform_action(states, actions):
        return skill_policy.get_action(
            torch.concatenate([states, actions], axis=-1), noisy=True
        )

    # Setup networks
    def get_q_network(observation_dim, action_dim):
        q_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim + action_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

        return q_network

    sparsity_lambda = 0.01

    def get_policy_network(observation_dim, action_dim):
        policy_network = HRLPolicy(
            observation_dim, action_dim, sparsity_loss_factor=sparsity_lambda
        )
        return policy_network

    q_optimizer_kwargs = {'lr': 1e-5}
    policy_optimizer_kwargs = {'lr': 1e-5}

    # Setup agent
    hrl_agent = SAC(
        state_dim=observation_dims,
        action_dim=num_skills,
        policy_class=get_policy_network,
        q_network_class=get_q_network,
        alpha=0.05,
        action_low=np.zeros(num_skills, dtype=np.float32),
        action_high=np.ones(num_skills, dtype=np.float32),
        policy_optimizer_kwargs=policy_optimizer_kwargs,
        q_optimizer_kwargs=q_optimizer_kwargs,
        device=device,
        log_writer=log_writer,
    )

    if model_load_path is not None:
        print('Loading model from ' + model_load_path)
        hrl_agent.load_checkpoint(model_load_path)

    visualize_robosuite(
        environment_name,
        robots,
        steps_per_episode,
        hrl_agent,
        device,
        action_transform_func=transform_action,
        output_folder=video_output_folder,
        output_name_prefix=f'{video_file_prefix}',
        config=config,
    )


def checkpoint_check(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file '{filepath}' not found")

    checkpoint = torch.load(filepath, map_location='cuda')
    print('Checkpoint contents:', checkpoint)  # Add this line to debug
    # self.set_state_dict(checkpoint)

    # logger.info(f'Checkpoint loaded from {filepath}')


if __name__ == '__main__':
    # Read config file for all settings
    current_dir = str(Path(__file__).parent.resolve())
    print('Current directory:', current_dir)

    with open(current_dir + '/config/ur5e_config_hrl_2.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # exit()
    environment_name = config['params']['environment_name']
    robots = config['params']['robots']
    num_envs = config['params']['num_envs']
    num_steps = config['evaluation_params']['num_steps']
    num_skills = config['params']['num_skills']

    # Folder paths
    skill_policy_path = config['training_params']['skill_policy_path']
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
        config=config,
    )
