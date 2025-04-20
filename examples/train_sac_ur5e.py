import time
from typing import Optional
import yaml
import datetime


import os


import torch
from torch.utils.tensorboard import SummaryWriter


from DIAYN import SAC, ReplayBuffer, make_env, rollout

# Robosuite stuff:


from gymnasium.vector import SyncVectorEnv

from pathlib import Path

from DIAYN.utils import (
    replay_post_processor,
)


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

    # Setup Networks
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

    q_optimizer_kwargs = {'lr': 1e-3}
    policy_optimizer_kwargs = {'lr': 1e-4}

    # Setup agent
    hrl_agent = SAC(
        state_dim=observation_dims,
        action_dim=action_dims,
        policy_class=get_policy_network,
        q_network_class=get_q_network,
        alpha=0.05,
        action_low=action_low,
        action_high=action_high,
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
    with open(current_dir + '/config/ur5e_config_sac.yaml', 'r') as file:
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
        # model_load_path=model_load_path,
        log_path=log_path,
        model_save_folder=model_save_folder,
        config=config,
    )
