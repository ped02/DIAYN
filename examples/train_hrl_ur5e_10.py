import time
from typing import Optional
import yaml
import datetime

import numpy as np

import os


import torch
from torch.utils.tensorboard import SummaryWriter


from DIAYN import SAC, DIAYNAgent, ReplayBuffer, make_env, rollout

# Robosuite stuff:


from gymnasium.vector import SyncVectorEnv

from pathlib import Path

from DIAYN.utils import (
    replay_post_processor,
)


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

    diayn_agent.load_checkpoint(model_load_path)

    diayn_agent.freeze()

    return diayn_agent


class HRLPolicy(torch.nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        skill_dim,
        sparsity_loss_factor: float = 1.0,
        smoothness_loss_factor: float = 1.0,
        modifier_loss_factor: float = 1.0,
        **kwargs,
    ):
        super(HRLPolicy, self).__init__(**kwargs)

        self.skill_selector = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2 * (skill_dim + action_dim)),
        )

        self.register_buffer(
            'sparsity_loss_factor', torch.tensor(sparsity_loss_factor)
        )
        self.register_buffer(
            'smoothness_loss_factor', torch.tensor(smoothness_loss_factor)
        )
        self.register_buffer(
            'modifier_loss_factor', torch.tensor(modifier_loss_factor)
        )
        self.register_buffer(
            'action_dim', torch.tensor(action_dim, dtype=torch.int64)
        )
        self.register_buffer(
            'skill_dim', torch.tensor(skill_dim, dtype=torch.int64)
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict, assign)

    def forward(self, states):
        value_mean, value_log_std = self.skill_selector(states).chunk(2, dim=-1)

        skill_mean, action_mean = torch.split(
            value_mean, (self.skill_dim, self.action_dim), dim=-1
        )
        skill_mean = torch.softmax(skill_mean, dim=-1)
        action_mean = torch.tanh(action_mean)

        return torch.concatenate(
            [skill_mean, action_mean, value_log_std], dim=-1
        )

    # def get_loss(self, states, actions, next_states):
    #     current_skill_mean, _ = self.skill_selector(states).chunk(2, dim=-1)
    #     next_skill_mean, _ = self.skill_selector(next_states).chunk(2, dim=-1)

    #     current_skill_mean = torch.softmax(current_skill_mean, dim=-1)
    #     next_skill_mean = torch.softmax(next_skill_mean, dim=-1)

    #     return self.smoothness_loss_factor * torch.mean(
    #         (next_skill_mean - current_skill_mean) ** 2
    #     )
    def get_loss(self, states, actions, next_states):
        value_mean, value_log_std = self.skill_selector(states).chunk(2, dim=-1)

        skill_mean, action_mean = torch.split(
            value_mean, (self.skill_dim, self.action_dim), dim=-1
        )

        return self.modifier_loss_factor * torch.mean(
            torch.sum(action_mean**2, dim=-1)
        )


def main(
    environment_name: str,
    robots: str,
    num_envs: int,
    episodes: int,
    steps_per_episode: int,
    num_skills: int,
    skill_policy_path: str,
    model_load_path: Optional[str] = None,
    log_path: Optional[str] = None,
    model_save_folder: Optional[str] = None,
    plot_dpi: float = 150.0,
    plot_trajectories: int = 5,
    plot_train_steps_period: Optional[int] = 1500,
    config: Optional[dict] = None,
):
    device = torch.device('cuda')
    print(f'Using device: {device}')

    # Setup logging
    log_writer = None if log_path is None else SummaryWriter(log_path)

    envs = SyncVectorEnv([make_env(config) for _ in range(num_envs)])

    observation_dims = envs.observation_space.shape[1]
    print('Observation dims: ', observation_dims)
    action_dims = envs.action_space.shape[1]

    action_low = envs.action_space.low[0]
    action_high = envs.action_space.high[0]

    # Replay buffer
    replay_buffer_size = 1_000_000
    replay_buffer = ReplayBuffer(
        replay_buffer_size,
        post_processor=lambda x: replay_post_processor(x, device),
    )

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

    action_low_device = torch.tensor(action_low, device=device)
    action_high_device = torch.tensor(action_high, device=device)

    def transform_action(states, actions):
        skill_vector, modify_action_vector = torch.split(
            actions, (num_skills, action_dims), dim=-1
        )

        skill_action = skill_policy.get_action(
            torch.concatenate([states, skill_vector], axis=-1), noisy=False
        )

        return torch.clamp(
            skill_action + modify_action_vector,
            min=action_low_device,
            max=action_high_device,
        )

    # Setup networks
    def get_q_network(observation_dim, _):
        q_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim + num_skills + action_dims, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

        return q_network

    sparsity_lambda = 0.01
    smoothness_lambda = 0.05
    modifier_lambda = 5.0

    def get_policy_network(observation_dim, _):
        policy_network = HRLPolicy(
            observation_dim,
            action_dims,
            num_skills,
            sparsity_loss_factor=sparsity_lambda,
            smoothness_loss_factor=smoothness_lambda,
            modifier_loss_factor=modifier_lambda,
        )
        return policy_network

    q_optimizer_kwargs = {'lr': 1e-3}
    policy_optimizer_kwargs = {'lr': 1e-4}

    # Setup agent
    hrl_agent_action_low = np.zeros(num_skills + action_dims, dtype=np.float32)
    hrl_agent_action_low[num_skills:] = -1.0

    hrl_agent_action_high = np.ones(num_skills + action_dims, dtype=np.float32)

    hrl_agent = SAC(
        state_dim=observation_dims,
        action_dim=num_skills,
        policy_class=get_policy_network,
        q_network_class=get_q_network,
        alpha=0.05,
        action_low=hrl_agent_action_low,
        action_high=hrl_agent_action_high,
        policy_optimizer_kwargs=policy_optimizer_kwargs,
        q_optimizer_kwargs=q_optimizer_kwargs,
        policy_gradient_clip_norm=1e-2,
        q_gradient_clip_norm=1e-2,
        device=device,
        log_writer=log_writer,
    )

    if model_load_path is not None:
        print('Loading model from ' + model_load_path)
        hrl_agent.load_checkpoint(model_load_path)

    # Training
    total_steps = 0

    def plot_skill_trajectories_phase():
        # TODO: Add this back for plotting later
        pass
        # skill_trajectory_visual = plot_skill_trajectories(
        #     environment_name,
        #     plot_trajectories,
        #     num_steps,
        #     diayn_agent,
        #     plot_dpi,
        # )
        # phase_skill_visual = plot_phase_skill(
        #     num_skills, diayn_agent, dpi=plot_dpi
        # )

        # log_writer.add_image(
        #     'Skill/Trajectory',
        #     image_numpy_to_torch(skill_trajectory_visual),
        #     total_steps,
        # )
        # log_writer.add_image(
        #     'Skill/Phase', image_numpy_to_torch(phase_skill_visual), total_steps
        # )

    def training_function(step):
        nonlocal total_steps
        if len(replay_buffer.buffer) > min(
            num_steps * num_skills * num_envs * 10, 10000
        ):  # at least 10 demonstrations of each randomly before start training
            hrl_agent.update(
                replay_buffer,
                step=total_steps,
                q_train_iterations=config['training_params'][
                    'q_train_iterations'
                ],
                policy_train_iterations=config['training_params'][
                    'policy_train_iterations'
                ],
                batch_size=64,
            )

            total_steps += 1

            if plot_train_steps_period is not None and log_writer is not None:
                if (total_steps + 1) % plot_train_steps_period == 0:
                    plot_skill_trajectories_phase()

    # Pre training spaces
    if log_writer is not None:
        plot_skill_trajectories_phase()

    print(
        '------------------------------- Beginning training -------------------------------'
    )
    start_time = time.time()
    for episode in range(episodes):
        if (episode) % 10 == 0 and episode > 0:
            print(
                f'Starting {episode} / {episodes} @ {time.time() - start_time:.3f} sec'
            )

            if model_save_folder is not None:
                print('Saving model states at episode ' + str(episode))
                model_save_path = os.path.join(
                    model_save_folder, ('episode_' + str(episode) + '.pt')
                )
                hrl_agent.save_checkpoint(model_save_path)

        mean_step_reward = rollout(
            envs,
            num_steps=steps_per_episode,
            replay_buffer=replay_buffer,
            agent=hrl_agent,
            device=device,
            reward_scale=1e2,
            action_transform_func=transform_action,
            post_step_func=training_function,
        )

        mean_step_reward  # surpress unused
        if log_writer is not None:
            log_writer.add_scalar(
                'stats/ avg reward', mean_step_reward, global_step=total_steps
            )

    # Post training spaces
    if log_writer is not None:
        plot_skill_trajectories_phase()

    # Save model
    if model_save_path is not None:
        print('Saving final model states')
        hrl_agent.save_checkpoint(model_save_path)


if __name__ == '__main__':
    # Read config file for all settings
    # Print current directory

    # Get location of this file
    current_dir = str(Path(__file__).parent.resolve())
    print('Current directory:', current_dir)
    with open(current_dir + '/config/ur5e_config_hrl_10.yaml', 'r') as file:
        config = yaml.safe_load(file)

    environment_name = config['params']['environment_name']
    robots = config['params']['robots']
    num_envs = config['params']['num_envs']
    num_steps = config['training_params']['num_steps']
    num_skills = config['params']['num_skills']
    episodes = config['training_params']['episodes']
    log_parent_folder = config['training_params'][
        'log_parent_folder'
    ]  # log path for tensorboard

    # model_load_path = config['training_params']['model_load_path']  # load path for tensorboard

    skill_policy_path = config['training_params']['skill_policy_path']

    # Setup for saving weights
    model_save_folder = config['training_params']['model_save_folder']
    run_name = 'run_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_save_folder = os.path.join(model_save_folder, run_name)
    if model_save_folder is not None:
        os.makedirs(model_save_folder, exist_ok=True)

    log_path = os.path.join(log_parent_folder, run_name)

    # Print all params in an orderly fashion:
    print(
        '------------------------------- Start DIAYN Training Parameters -------------------------------'
    )
    print('Environment name: ', environment_name)
    print('Robots: ', robots)
    print('Number of environments: ', num_envs)
    print('Number of skills: ', num_skills)
    print(' --------------------')
    print('Number of episodes: ', episodes)
    print('Number of steps: ', num_steps)
    print(
        'q_train_iterations: ', config['training_params']['q_train_iterations']
    )
    print(
        'policy_train_iterations: ',
        config['training_params']['policy_train_iterations'],
    )
    print(' --------------------')
    print('Log path: ', log_path)
    print('Model save folder: ', model_save_folder)

    # Show which observations we are using:
    use_eef_state = config['observations']['use_eef_state']
    use_joint_vels = config['observations']['use_joint_vels']
    use_cube_pos = config['observations']['use_cube_pos']
    use_vae = config['observations']['use_vae']

    print(' --------------------')
    print('Observations used:')
    if use_vae:
        print('     VAE latent space')
    else:
        if use_eef_state:
            print('     End effector state')
        if use_joint_vels:
            print('     Joint velocities')
        if use_cube_pos:
            print('     Cube position')

    print(
        '------------------------------- End DIAYN Training Parameters -------------------------------'
    )

    # TODO: Print which observation states we are going to use

    main(
        environment_name,
        robots,
        num_envs,
        episodes,
        num_steps,
        num_skills,
        skill_policy_path,
        # model_load_path=model_load_path,
        log_path=log_path,
        model_save_folder=model_save_folder,
        config=config,
    )
