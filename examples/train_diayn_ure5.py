import time
from typing import Optional
import yaml
import datetime

import numpy as np

import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from DIAYN import ReplayBuffer, DIAYNAgent, make_env, rollout_skill

# Robosuite stuff:


from gymnasium.vector import SyncVectorEnv

from pathlib import Path

from DIAYN.utils import (
    replay_post_processor,
    pad_to_dim_2,
    plot_to_image,
)


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

    # Setup networks
    def get_q_network(observation_dim, action_dim):
        q_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim + action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

        return q_network

    def get_policy_network(observation_dim, action_dim):
        policy_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
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

    q_optimizer_kwargs = {'lr': 1e-4}
    discriminator_optimizer_kwargs = {'lr': 4e-4}
    policy_optimizer_kwargs = {'lr': 3e-5}

    constant_alpha = False

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
        print('Loading model from ' + model_load_path)
        diayn_agent.load_checkpoint(model_load_path)

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
            diayn_agent.update(
                replay_buffer,
                step=total_steps,
                q_train_iterations=config['training_params'][
                    'q_train_iterations'
                ],
                policy_train_iterations=config['training_params'][
                    'policy_train_iterations'
                ],
                discriminator_train_iterations=config['training_params'][
                    'discriminator_train_iterations'
                ],
                batch_size=64,
                constant_alpha=constant_alpha,
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

    # Get location of this file
    current_dir = str(Path(__file__).parent.resolve())
    print('Current directory:', current_dir)
    with open(current_dir + '/config/ur5e_config.yaml', 'r') as file:
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

    model_load_path = config['training_params'][
        'model_load_path'
    ]  # load path for tensorboard

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
    print(
        'discriminator_train_iterations: ',
        config['training_params']['discriminator_train_iterations'],
    )
    print(' --------------------')
    print('Log path: ', log_path)
    print('Model save folder: ', model_save_folder)
    print('Model load path: ', model_load_path)

    # Show which observations we are using:
    use_eef_state = config['observations']['use_eef_state']
    use_joint_vels = config['observations']['use_joint_vels']
    use_cube_pos = config['observations']['use_cube_pos']

    print(' --------------------')
    print('Observations used:')
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
        model_load_path=model_load_path,
        log_path=log_path,
        model_save_folder=model_save_folder,
        config=config,
    )
