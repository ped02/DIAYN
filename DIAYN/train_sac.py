import argparse
from typing import Optional


import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from DIAYN import ReplayBuffer, SAC, visualize, rollout


def main(environment_name: str, render: bool, seed: Optional[int] = None):
    # render_mode = None
    # if render:
    #     render_mode = 'human'

    # env = gym.make(environment_name, render_mode=render_mode)

    # if seed is not None:
    #     env.reset(seed=seed)

    episodes = 512
    n_envs = 32
    n_steps = 512

    envs = gym.make_vec(
        environment_name,
        vectorization_mode=gym.VectorizeMode.SYNC,
        num_envs=n_envs,
    )

    device = torch.device('cuda')

    def replay_post_processor(samples):
        return [torch.stack(e).to(device) for e in zip(*samples)]

    replay_buffer_size = 1_000_000
    replay_buffer = ReplayBuffer(
        replay_buffer_size, post_processor=replay_post_processor
    )

    observation_dims = envs.observation_space.shape[1]
    action_dims = envs.action_space.shape[1]

    action_low = envs.action_space.low[0]
    action_high = envs.action_space.high[0]

    ## Assume continuous control only
    print(f'{observation_dims=} {action_dims=}')
    # print(f'{type(envs.action_space)=}')
    # print(f'{envs.action_space.low=}')
    # print(f'{envs.action_space.high=}')

    def get_q_network(observation_dim, action_dim):
        q_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim + action_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

        return q_network

    def get_policy_network(observation_dim, action_dim):
        policy_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2 * action_dim),
        )
        return policy_network

    optimizer_kwargs = {'lr': 3e-4}

    log_writer = SummaryWriter('runs/test_10')

    sac_agent = SAC(
        observation_dims,
        action_dims,
        get_policy_network,
        get_q_network,
        alpha=0.1,
        action_low=action_low,
        action_high=action_high,
        policy_optimizer_kwargs=optimizer_kwargs,
        q_optimizer_kwargs=optimizer_kwargs,
        log_writer=log_writer,
        device=device,
    )

    for episode in range(episodes):
        if (episode + 1) % 50 == 0:
            print(f'Starting {episode+1}/{episodes}')

        # Roll out
        sac_agent.pre_episode()
        mean_step_reward = rollout(
            envs,
            num_steps=n_steps,
            replay_buffer=replay_buffer,
            agent=sac_agent,
            device=device,
            reward_scale=0.01,
        )
        # observations_raw, info = envs.reset()
        # observations = torch.Tensor(observations_raw)

        # total_reward = torch.zeros(n_envs)

        # for step in range(n_steps):

        #     with torch.no_grad():
        #         actions = sac_agent.get_action(observations.to(device), noisy=True).cpu()
        #     next_observations_raw, rewards_raw, terminated, truncated, _ = envs.step(actions.numpy())

        #     next_observations = torch.Tensor(next_observations_raw)
        #     rewards = torch.Tensor(rewards_raw)

        #     replay_buffer.add((observations, actions, rewards * 0.01, next_observations, 1 - torch.Tensor(terminated | truncated)))
        #     observations = next_observations

        #     total_reward += rewards

        # log_writer.add_scalar('stats/Rewards', total_reward.mean().item() / n_steps, episode)
        log_writer.add_scalar(
            'stats/Rewards', mean_step_reward / n_steps, episode
        )

        # Train
        sac_agent.update(
            replay_buffer,
            step=episode,
            q_train_iterations=64,
            policy_train_iterations=4,
            batch_size=32,
        )

        # observation, info = env.reset()

        # episode_over = False
        # for _ in range(1000):
        #     action = env.action_space.sample()  # agent policy that uses the observation and info
        #     observation, reward, terminated, truncated, info = env.step(action)

        #     episode_over = terminated or truncated
        #     if episode_over:
        #         env.reset()

    envs.close()

    # Visualize
    visualize(environment_name, 512, sac_agent, device)

    # env = gym.make(environment_name, render_mode='rgb_array')

    # env_recorder = gym.wrappers.RecordVideo(env, video_folder='./', episode_trigger=lambda x:True)

    # observations_raw, _ = env_recorder.reset()

    # for t in range(512):
    #     with torch.no_grad():
    #         actions = sac_agent.get_action(torch.Tensor(observations_raw).to(device)[None, :])[:,0]
    #     observations_raw, rewards, done, _, _ = env_recorder.step(actions.cpu().numpy())
    #     if done:
    #         break

    # env_recorder.close()

    log_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('env_name', type=str, help='Name of environment')
    parser.add_argument(
        '--render', action='store_true', help='Run environment in render mode'
    )
    parser.add_argument('--seed', type=int, default=123, help='Reset seed')

    args = parser.parse_args()

    main(environment_name=args.env_name, render=args.render)
