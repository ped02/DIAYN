import time
from typing import Optional
import yaml

import numpy as np

import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from DIAYN import ReplayBuffer, DIAYNAgent, rollout_skill

# Robosuite stuff:
import argparse

import robosuite as suite
from robosuite import load_composite_controller_config

from gymnasium.vector import SyncVectorEnv
from robosuite.wrappers import GymWrapper

from DIAYN.utils import (
    replay_post_processor,
    pad_to_dim_2,
    plot_to_image,
)


def make_env(env_name, robots):
    def _thunk():
        print('Creating new environment')

        controller_config = load_composite_controller_config(
            controller=None,
            robot='Panda',
        )

        config = {
            'env_name': env_name,
            'robots': robots,
            'controller_configs': controller_config,
        }

        robosuite_env = suite.make(
            **config,
            has_renderer=False,
            has_offscreen_renderer=False,
            render_camera='agentview',
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=False,
        )
        env = GymWrapper(robosuite_env)

        # Ensure metadata exists and is a dict before modifying
        if env.metadata is None:
            env.metadata = {}
        env.metadata['render_modes'] = []
        env.metadata['autoreset'] = False

        return env

    return _thunk


def plot_skill_trajectories(
    environment_name: str,
    num_trajectories: int,
    steps_per_episode: int,
    agent: DIAYNAgent,
    dpi: Optional[float] = None,
):
    env = gym.make(environment_name)

    # Rollout
    skill_trajectories = []
    for z in range(num_skills):
        skill_index = torch.tensor(z).reshape(
            1,
        )
        skill_vector = torch.nn.functional.one_hot(
            skill_index, num_classes=num_skills
        )
        expanded_skill_vector = skill_vector.expand(1, -1)

        skill_samples = []
        for t in range(num_trajectories):
            sample = []
            observations_raw, info = env.reset()
            observations = torch.cat(
                [
                    pad_to_dim_2(torch.Tensor(observations_raw)),
                    expanded_skill_vector,
                ],
                dim=-1,
            )

            # sample.append(observations_raw.tolist())
            sample.append([*observations_raw.tolist(), 0.0])

            for step in range(steps_per_episode):
                with torch.no_grad():
                    actions = agent.get_action(
                        observations.to(agent.device), noisy=True
                    ).cpu()
                (
                    next_observations_raw,
                    rewards_raw,
                    terminated,
                    truncated,
                    _,
                ) = env.step(actions.numpy())

                next_observations = torch.cat(
                    [
                        pad_to_dim_2(torch.Tensor(next_observations_raw)),
                        expanded_skill_vector,
                    ],
                    dim=-1,
                )

                with torch.no_grad():
                    discriminator_logits = agent.discriminator(
                        pad_to_dim_2(torch.Tensor(next_observations_raw)).to(
                            agent.device
                        )
                    )
                psuedo_reward = torch.nn.functional.log_softmax(
                    discriminator_logits, dim=-1
                ).gather(
                    -1, skill_index.unsqueeze(0).to(agent.device)
                ) + torch.log(
                    torch.tensor(agent.skill_dim, device=agent.device)
                )  # Assume uniform distribution

                observations = next_observations

                # sample.append(next_observations_raw.tolist())
                sample.append(
                    [*next_observations_raw.tolist(), psuedo_reward.item()]
                )

            skill_samples.append(sample)

        skill_trajectories.append(skill_samples)

    skills_trajectories_np = np.array(skill_trajectories)

    colors = plt.get_cmap('tab20c', num_skills)

    # Create plot
    fig = plt.figure()

    for skill_id in range(num_skills):
        for traj_id in range(num_trajectories):  # Iterate over trajectories
            trajectory = skills_trajectories_np[
                skill_id, traj_id
            ]  # Extract (16, 2) trajectory
            plt.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=colors(skill_id),
                label=f'Skill {skill_id}' if traj_id == 0 else '',
            )

    fig.gca().set_xlim((-3, 3))
    fig.gca().set_ylim((-3, 3))

    fig.gca().set_xlabel('X')
    fig.gca().set_ylabel('Y')
    fig.gca().set_title('Trajectories Colored by Skill ID')
    fig.legend()

    # Plt to image
    plot_image = plot_to_image(fig, dpi=dpi)
    plt.close(fig)

    return plot_image


def plot_phase_skill(
    num_skills: int,
    agent: DIAYNAgent,
    resolution=0.25,
    dpi: Optional[float] = None,
):
    # Define a consistent colormap
    colors = plt.get_cmap(
        'tab20c', num_skills
    )  # Use categorical colormap with exactly k colors
    cmap = ListedColormap(
        [colors(i) for i in range(num_skills)]
    )  # Ensure colors are fixed

    # Define normalization for consistent color mapping
    norm = BoundaryNorm(np.arange(-0.5, num_skills, 1), num_skills)

    points = np.arange(-3.0, 3.0, resolution)  # bounds from environment [-3, 3]
    grid = np.stack(np.meshgrid(points, points), -1)

    with torch.no_grad():
        result = (
            agent.discriminator(
                torch.Tensor(grid).to(agent.device).reshape(-1, 2)
            )
            .cpu()
            .numpy()
        )

    skill_classification = np.argmax(result, axis=-1)
    skill_grid = skill_classification.reshape(grid.shape[:-1])
    fig = plt.figure()
    plt.imshow(
        skill_grid,
        extent=[-3.0, 3.0, -3.0, 3.0],
        origin='lower',
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(ticks=np.arange(num_skills))

    plot_image = plot_to_image(fig, dpi=dpi)
    plt.close(fig)

    return plot_image


def main(
    environment_name: str,
    robots: str,
    num_envs: int,
    episodes: int,
    steps_per_episode: int,
    num_skills: int,
    log_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
    plot_dpi: float = 150.0,
    plot_trajectories: int = 5,
    plot_train_steps_period: Optional[int] = 1500,
):
    device = torch.device('cuda')
    print(f'Using device: {device}')

    # Setup logging
    log_writer = None if log_path is None else SummaryWriter(log_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=str, default='Lift')
    parser.add_argument(
        '--robots',
        nargs='+',
        type=str,
        default='Panda',
        help='Which robot(s) to use in the env',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='default',
        help='Specified environment configuration if necessary',
    )
    parser.add_argument(
        '--arm',
        type=str,
        default='right',
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        '--switch-on-grasp',
        action='store_true',
        help='Switch gripper control on gripper action',
    )
    parser.add_argument(
        '--toggle-camera-on-grasp',
        action='store_true',
        help='Switch camera angle on gripper action',
    )
    parser.add_argument(
        '--controller',
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples) or None to get the robot's default controller if it exists",
    )
    parser.add_argument('--device', type=str, default='keyboard')
    parser.add_argument(
        '--pos-sensitivity',
        type=float,
        default=1.0,
        help='How much to scale position user inputs',
    )
    parser.add_argument(
        '--rot-sensitivity',
        type=float,
        default=1.0,
        help='How much to scale rotation user inputs',
    )
    parser.add_argument(
        '--max_fr',
        default=20,
        type=int,
        help='Sleep when simluation runs faster than specified frame rate; 20 fps is real time.',
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    # Create argument configuration
    config = {
        'env_name': args.environment,
        'robots': args.robots,
        'controller_configs': controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if 'TwoArm' in args.environment:
        config['env_configuration'] = args.config
    else:
        args.config = None

    envs = SyncVectorEnv(
        [make_env(environment_name, robots) for _ in range(num_envs)]
    )

    observation_dims = envs.observation_space.shape[1]
    action_dims = envs.action_space.shape[1]

    action_low = envs.action_space.low[0]
    action_high = envs.action_space.high[0]

    # Replay buffer
    replay_buffer_size = 1_000_000
    replay_buffer = ReplayBuffer(
        replay_buffer_size,
        post_processor=lambda x: replay_post_processor(x, device),
    )

    # Setup networks
    def get_q_network(observation_dim, action_dim):
        q_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim + action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
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
            torch.nn.Linear(256, 256),
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
        if (
            len(replay_buffer.buffer) > num_steps * num_skills * num_envs * 10
        ):  # at least 10 demonstrations of each randomly before start training
            diayn_agent.update(
                replay_buffer,
                step=total_steps,
                q_train_iterations=1,
                policy_train_iterations=1,
                discriminator_train_iterations=1,
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
        if (episode + 1) % 50 == 0:
            print(
                f'Starting {episode + 1} / {episodes} @ {time.time() - start_time:.3f} sec'
            )

            if model_save_path is not None:
                print('Saving model states at episode ' + str(episode + 1))
                diayn_agent.save_checkpoint(model_save_path)
        skill_index = torch.randint(0, num_skills, (1,))
        skill_vector = torch.nn.functional.one_hot(
            skill_index, num_classes=num_skills
        )
        mean_step_reward = rollout_skill(
            envs,
            num_steps=steps_per_episode,
            replay_buffer=replay_buffer,
            agent=diayn_agent,
            device=device,
            reward_scale=0.01,  # Not used
            skill_index=skill_index,
            skill_vector=skill_vector,
            post_step_func=training_function,
        )

        mean_step_reward  # surpress unused

    # Post training spaces
    if log_writer is not None:
        plot_skill_trajectories_phase()

    # Save model
    if model_save_path is not None:
        print('Saving final model states')
        diayn_agent.save_checkpoint(model_save_path)


if __name__ == '__main__':
    # Read config file for all settings
    # Print current directory
    print('Current working directory:', os.getcwd())
    with open('./config/ur5e_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    environment_name = config['params']['environment_name']
    robots = config['params']['robots']
    num_envs = config['params']['num_envs']
    num_steps = config['training_params']['num_steps']
    num_skills = config['params']['num_skills']
    episodes = config['training_params']['episodes']
    log_path = config['training_params']['log_path']  # log path for tensorboard

    # Setup for saving weights
    model_save_folder = config['training_params']['model_save_folder']
    if model_save_folder is not None:
        os.makedirs(model_save_folder, exist_ok=True)

    idx = 0
    while os.path.exists(model_save_folder + '/' + str(idx) + '.pt'):
        idx += 1
    model_save_path = model_save_folder + '/' + str(idx) + '.pt'

    # Print all params in an orderly fashion:
    print(
        '------------------------------- DIAYN Training Parameters -------------------------------'
    )
    print('Environment name: ', environment_name)
    print('Robots: ', robots)
    print('Number of environments: ', num_envs)
    print('Number of steps: ', num_steps)
    print('Number of skills: ', num_skills)
    print('Number of episodes: ', episodes)
    print('Log path: ', log_path)
    print('Model save folder: ', model_save_folder)
    print('Model save path: ', model_save_path)

    main(
        environment_name,
        robots,
        num_envs,
        episodes,
        num_steps,
        num_skills,
        log_path=log_path,
        model_save_path=model_save_path,
    )
