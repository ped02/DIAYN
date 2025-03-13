import time
from typing import Optional
import os

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from DIAYN import ReplayBuffer, DIAYNAgent, rollout_skill, evaluate_agent
from DIAYN.utils import (
    replay_post_processor,
)

DIAYN_PARAMS = {
    'layer_size': 300,
    'batch_size': 128,
    'num_steps': 1000,
    'num_epochs': 10000,
    'lr': 3e-4,
}

# SAC_PARAMS = {
#     'Hopper-v5': {
#         'layer_size': 128,
#         'lr': 3e-4,
#         'discount': 0.99,
#         'batch_size': 128,
#         'num_steps': 1000,
#         'num_epochs': 3000,
#     },
# }




def main(
    environment_name: str,
    num_envs: int,
    episodes: int,
    steps_per_episode: int,
    num_skills: int,
    log_path: Optional[str] = None,
    model_save_folder: Optional[str] = None,
    plot_dpi: float = 150.0,
    plot_trajectories: int = 5,
    plot_train_steps_period: Optional[int] = 15000,
    evaluate_episodes: int = 10,
    model_load_path: Optional[str] = None,
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
        layer_size = DIAYN_PARAMS['layer_size']
        q_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim + action_dim, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, 1),
        )

        return q_network

    def get_policy_network(observation_dim, action_dim):
        layer_size = DIAYN_PARAMS['layer_size']
        policy_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, 2 * action_dim),
        )
        return policy_network

    def get_discriminiator_network(observation_dim, skill_dim):
        # Output logits
        layer_size = DIAYN_PARAMS['layer_size']
        discriminiator_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, skill_dim),
        )

        return discriminiator_network

    q_optimizer_kwargs = {'lr': DIAYN_PARAMS['lr']}
    discriminator_optimizer_kwargs = {'lr': 4e-4}
    policy_optimizer_kwargs = {'lr': DIAYN_PARAMS['lr']}

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
    
    # Training
    total_steps = 0

    def plot_reward_histogram():
        total_returns = []
        mean_step_reward = []
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
            mean_step_reward.append(
                np.nanmean(
                    result_dict['total_return'] / result_dict['episode_length']
                )
            )
        log_writer.add_histogram(
            'Stats/Skill Total Return', np.array(total_returns), total_steps
        )
        log_writer.add_histogram(
            'Stats/Skill Average Reward',
            np.array(mean_step_reward),
            total_steps,
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
                batch_size=DIAYN_PARAMS['batch_size'],
            )

            total_steps += 1

            if plot_train_steps_period is not None and log_writer is not None:
                if (total_steps + 1) % plot_train_steps_period == 0:
                    plot_reward_histogram()

    # Pre training spaces
    if log_writer is not None:
        plot_reward_histogram()

    start_time = time.time()
    for episode in range(episodes):
        if (episode + 1) % 50 == 0:
            print(
                f'Starting {episode + 1} / {episodes} @ {time.time() - start_time:.3f} sec'
            )
            # Save model
            if model_save_folder is not None:
                diayn_agent.save_checkpoint(model_save_folder + f'/checkpoint_{episode + 1}.pt')
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
        plot_reward_histogram()

    # Save model
    if model_save_folder is not None:
        diayn_agent.save_checkpoint(model_save_folder + '_final.pt')


if __name__ == '__main__':
    environment_name = 'Hopper-v5'

    episodes = DIAYN_PARAMS['num_epochs']
    num_envs = 4
    num_steps = DIAYN_PARAMS['num_steps']  # 1000
    num_skills = 50

    # Create new run folder for current run
    root_dir = 'hopper_runs'
    idx = 0
    while os.path.exists(root_dir + '/run_' + str(idx)):
        idx += 1
    
    root_dir = root_dir + '/run_' + str(idx)
    os.makedirs(root_dir, exist_ok=True)

    # Create log folder inside new run folder
    log_path = root_dir + '/log'
    os.makedirs(log_path, exist_ok=True)
    
    # Create model save folder inside new run folder
    model_save_folder = root_dir + "/weights"
    os.makedirs(model_save_folder, exist_ok=True)

    # Use model_load_path if we want
    use_pretrained = True

    if use_pretrained:
        model_load_path = '/home/rmineyev3/DIAYN/examples/hopper_runs/run_3/weights/checkpoint_1250.pt'
    else:
        model_load_path = None
    
    
    # # look through folder, and set model name to be the next number
    # idx = 0
    # while os.path.exists(model_save_folder + '/' + str(idx) + '.pt'):
    #     idx += 1
    # model_save_path = model_save_folder + '/' + str(idx) + '.pt'

    print("Log path: ", log_path)
    print("Model save folder: ", model_save_folder)
    print("Model load path: ", model_load_path)

    main(
        environment_name,
        num_envs,
        episodes,
        num_steps,
        num_skills,
        log_path=log_path,
        model_save_folder=model_save_folder,
        model_load_path=model_load_path
    )
