import time
from typing import Optional

import numpy as np

import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from DIAYN import ReplayBuffer, DIAYNAgent, rollout_skill, evaluate_agent
from DIAYN.utils import (
    replay_post_processor,
    pad_to_dim_2,
    plot_to_image,
    image_numpy_to_torch,
)


def plot_skill_trajectories(
    environment_name: str,
    num_trajectories: int,
    steps_per_episode: int,
    agent: DIAYNAgent,
    dpi: Optional[float] = None,
    x_state_index=0,
    y_state_index=1,
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
                ) = env.step(actions.numpy()[0])

                # Im not sure why this env return next obs in wrong shape
                next_observations_raw = next_observations_raw.reshape(-1)
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

            y_values = trajectory[:, y_state_index]
            time_steps = np.arange(len(y_values))

            plt.plot(
                time_steps,
                y_values,
                color=colors(skill_id),
                label=f'Skill {skill_id}' if traj_id == 0 else '',
            )

    fig.gca().set_xlim(
        (
            0, 
            steps_per_episode
        )
    )
    fig.gca().set_ylim(
        (
            env.observation_space.low[y_state_index],
            env.observation_space.high[y_state_index],
        )
    )

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
    points_count: int = 25,
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

    x_points = np.linspace(-1.2, 0.6, points_count)
    y_points = np.linspace(-0.07, 0.07, points_count)
    grid = np.stack(np.meshgrid(x_points, y_points), -1)

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
        extent=[-1.2, 0.6, -0.07, 0.07],
        origin='lower',
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(ticks=np.arange(num_skills), orientation='horizontal')

    plot_image = plot_to_image(fig, dpi=dpi)
    plt.close(fig)

    return plot_image


def main(
    environment_name: str,
    num_envs: int,
    episodes: int,
    steps_per_episode: int,
    num_skills: int,
    log_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
    plot_dpi: float = 150.0,
    plot_trajectories: int = 5,
    plot_train_steps_period: Optional[int] = 15000,
    evaluate_episodes: int = 10,
):
    device = torch.device('cuda')
    print(f'Using device: {device}')

    # Setup logging
    log_writer = None if log_path is None else SummaryWriter(log_path)

    # Setup env variables
    envs = gym.make_vec(
        environment_name,
        vectorization_mode=gym.VectorizeMode.SYNC,
        num_envs=num_envs,
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

    # Training
    total_steps = 0

    def plot_skill_trajectories_phase():
        skill_trajectory_visual = plot_skill_trajectories(
            environment_name,
            plot_trajectories,
            num_steps,
            diayn_agent,
            plot_dpi,
        )
        phase_skill_visual = plot_phase_skill(
            num_skills, diayn_agent, dpi=plot_dpi
        )

        log_writer.add_image(
            'Skill/Trajectory',
            image_numpy_to_torch(skill_trajectory_visual),
            total_steps,
        )
        log_writer.add_image(
            'Skill/Phase', image_numpy_to_torch(phase_skill_visual), total_steps
        )

    def plot_reward_histogram():
        total_returns = []
        for z in range(num_skills):
            result_dict = evaluate_agent(
                environment_name,
                steps_per_episode,
                evaluate_episodes,
                diayn_agent,
                device=device,
                skill_index=z,
                num_skills=num_skills,
            )
            total_returns.append(np.mean(result_dict['total_return']))
        log_writer.add_histogram(
            'Stats/Skill Total Return', np.array(total_returns), total_steps
        )

    def training_function(step):
        nonlocal total_steps
        if len(replay_buffer.buffer) > min(
            num_steps * num_skills * num_envs * 10, 10000
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
                    plot_reward_histogram()

    # Pre training spaces
    if log_writer is not None:
        plot_skill_trajectories_phase()
        plot_reward_histogram()

    start_time = time.time()
    for episode in range(episodes):
        print(
            f'Starting {episode + 1} / {episodes} @ {time.time() - start_time:.3f} sec'
        )
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
        plot_reward_histogram()

    if model_save_path is not None:
        diayn_agent.save_checkpoint(model_save_path)


if __name__ == '__main__':
    environment_name = 'MountainCarContinuous-v0'

    episodes = 300
    num_envs = 5
    num_steps = 200  # 1000
    num_skills = 30

    log_path = 'runs/diayn_mountain_1'

    # Check if output folder exists. If not, create it
    model_save_folder = 'weights/mountain_car'
    if model_save_folder is not None:
        os.makedirs(model_save_folder, exist_ok=True)

    idx = 0
    while os.path.exists(model_save_folder + '/' + str(idx) + '.pt'):
        idx += 1
    model_save_path = model_save_folder + '/' + str(idx) + '.pt'
    print("Model save path: ", model_save_path)

    main(
        environment_name,
        num_envs,
        episodes,
        num_steps,
        num_skills,
        log_path=log_path,
        plot_train_steps_period=2000,
        model_save_path=model_save_path
    )
