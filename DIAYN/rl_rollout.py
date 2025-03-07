from typing import Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from DIAYN import AgentBase, ReplayBuffer
from DIAYN.utils import pad_to_dim_2

def rollout(
        environment: Union[gym.Env, gym.vector.VectorEnv],
        num_steps: int,
        replay_buffer: ReplayBuffer,
        agent: AgentBase,
        device,
        reward_scale: float = 1.0
        ):
    """Rollout agent and save trajectory to replay buffer. Log reward once at the end of rollout
    
    Args:
        environment (gym.Env or gym.Vector.VectorEnv): Environment.
        num_steps (int): Number of steps to rollout for
        replay_buffer (ReplayBuffer): Replay buffer to save to
        agent (AgentBase): Agent to rollout
        device (str or pytorch device): Device of agent

    Store (observation, action, scaled reward, next observation, not done)
    """
    
    n_envs = 1
    if isinstance(environment, gym.vector.VectorEnv):
        n_envs = environment.num_envs

    observations_raw, info = environment.reset()
    observations = pad_to_dim_2(torch.Tensor(observations_raw))

    total_reward = torch.zeros(n_envs)

    for step in range(num_steps):

        with torch.no_grad():
            actions = agent.get_action(observations.to(device), noisy=True).cpu()
        next_observations_raw, rewards_raw, terminated, truncated, _ = environment.step(actions.numpy())

        next_observations = pad_to_dim_2(torch.Tensor(next_observations_raw))

        rewards = pad_to_dim_2(torch.Tensor(rewards_raw))

        replay_buffer.add((observations, actions, reward_scale * rewards, next_observations, 1 - torch.Tensor(terminated | truncated)), split_first_dim=True)
        observations = next_observations

        total_reward += reward_scale * rewards.squeeze()
    
    mean_step_reward = total_reward.mean().item()/num_steps

    return mean_step_reward

def rollout_skill(
        environment: Union[gym.Env, gym.vector.VectorEnv],
        num_steps: int,
        replay_buffer: ReplayBuffer,
        agent: AgentBase,
        device,
        skill_index,
        skill_vector,
        reward_scale: float = 1.0
        ):
    """Rollout agent and save trajectory to replay buffer. Log reward once at the end of rollout
    
    Args:
        environment (gym.Env or gym.Vector.VectorEnv): Environment.
        num_steps (int): Number of steps to rollout for
        replay_buffer (ReplayBuffer): Replay buffer to save to
        agent (AgentBase): Agent to rollout
        device (str or pytorch device): Device of agent
        skill_index (torch.Tensor int 1x): Skill index
        skill_vector (torch.Tensor int 1xnum_skill): One hot skill vector

    Store (observation, action, scaled reward, next observation, not done)
    """
    
    n_envs = 1
    if isinstance(environment, gym.vector.VectorEnv):
        n_envs = environment.num_envs

    expanded_skill_vector = skill_vector.expand(n_envs, -1)

    observations_raw, info = environment.reset()
    observations = torch.cat([pad_to_dim_2(torch.Tensor(observations_raw)), expanded_skill_vector], dim=-1)

    total_reward = torch.zeros(n_envs)

    skill_index_torch = skill_index.repeat(n_envs, 1)

    for step in range(num_steps):

        with torch.no_grad():
            actions = agent.get_action(observations.to(device), noisy=True).cpu()
        next_observations_raw, rewards_raw, terminated, truncated, _ = environment.step(actions.numpy())

        next_observations = torch.cat([pad_to_dim_2(torch.Tensor(next_observations_raw)), expanded_skill_vector], dim=-1)
        
        rewards = pad_to_dim_2(torch.Tensor(rewards_raw))

        replay_buffer.add((observations, skill_index_torch, actions, reward_scale * rewards, next_observations, 1 - torch.Tensor(terminated | truncated)), split_first_dim=True)
        observations = next_observations

        total_reward += reward_scale * rewards.squeeze()
    
    mean_step_reward = total_reward.mean().item()/num_steps

    return mean_step_reward