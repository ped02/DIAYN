from typing import Optional, Union, Callable

import torch

import gymnasium as gym

from DIAYN import AgentBase, ReplayBuffer
from DIAYN.utils import pad_to_dim_2


def rollout(
    environment: Union[gym.Env, gym.vector.VectorEnv],
    num_steps: int,
    replay_buffer: ReplayBuffer,
    agent: AgentBase,
    device,
    reward_scale: float = 1.0,
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
            actions = agent.get_action(
                observations.to(device), noisy=True
            ).cpu()
        (
            next_observations_raw,
            rewards_raw,
            terminated,
            truncated,
            _,
        ) = environment.step(actions.numpy())

        next_observations = pad_to_dim_2(torch.Tensor(next_observations_raw))

        rewards = pad_to_dim_2(torch.Tensor(rewards_raw), dim=1)

        not_dones = pad_to_dim_2(
            1 - torch.Tensor(terminated | truncated), dim=1
        )

        replay_buffer.add(
            (
                observations,
                actions,
                reward_scale * rewards,
                next_observations,
                not_dones,
            ),
            split_first_dim=True,
        )
        observations = next_observations

        total_reward += reward_scale * rewards.squeeze()

    mean_step_reward = total_reward.mean().item() / num_steps

    return mean_step_reward


def rollout_skill(
    environment: Union[gym.Env, gym.vector.VectorEnv],
    num_steps: int,
    replay_buffer: ReplayBuffer,
    agent: AgentBase,
    device,
    skill_index,
    skill_vector,
    reward_scale: float = 1.0,
    post_step_func: Optional[Callable[[int], None]] = None,
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
    observations = torch.cat(
        [
            pad_to_dim_2(torch.Tensor(observations_raw)).to(device),
            expanded_skill_vector.to(device),
        ],
        dim=-1,
    )

    total_reward = torch.zeros(n_envs)

    skill_index_torch = skill_index.repeat(n_envs, 1)

    for step in range(num_steps):
        with torch.no_grad():
            actions = agent.get_action(
                observations.to(device), noisy=True
            ).cpu()
        (
            next_observations_raw,
            rewards_raw,
            terminated,
            truncated,
            _,
        ) = environment.step(actions.numpy())

        next_observations = torch.cat(
            [
                pad_to_dim_2(torch.Tensor(next_observations_raw)).to(device),
                expanded_skill_vector.to(device),
            ],
            dim=-1,
        )

        rewards = pad_to_dim_2(torch.Tensor(rewards_raw), dim=1)

        not_dones = pad_to_dim_2(
            1 - torch.Tensor(terminated | truncated), dim=1
        )

        if replay_buffer is not None:
            replay_buffer.add(
                (
                    observations,
                    skill_index_torch,
                    actions,
                    reward_scale * rewards,
                    next_observations,
                    not_dones,
                ),
                split_first_dim=True,
            )
        observations = next_observations

        total_reward += reward_scale * rewards.squeeze()

        if post_step_func is not None:
            post_step_func(step)

    mean_step_reward = total_reward.mean().item() / num_steps

    return mean_step_reward


def rollout_hrl(
    environment: Union[gym.Env, gym.vector.VectorEnv],
    episode_length: int,
    skill_duration: int,
    agent: AgentBase,
    high_level_policy,
    device,
    num_skills: int,
    reward_scale: float = 1.0,
    log_writer=None,
    update_step: Optional[int] = None,
):
    n_envs = 1
    if isinstance(environment, gym.vector.VectorEnv):
        n_envs = environment.num_envs

    observations_raw, _ = environment.reset()
    observations = pad_to_dim_2(torch.Tensor(observations_raw)).to(device)

    total_reward = torch.zeros(n_envs, device=device)
    log_probs, returns = [], []

    for step in range(episode_length):
        if step % skill_duration == 0:
            state = observations[0]  # Assume single env
            skill_idx, log_prob_z = high_level_policy.select_skill(
                state, deterministic=False
            )
            skill_vec = torch.nn.functional.one_hot(
                torch.tensor(skill_idx, device=device), num_classes=num_skills
            ).unsqueeze(0)
            skill_vec = skill_vec.expand(n_envs, -1)
            log_probs.append(log_prob_z)
            step_reward = torch.zeros(n_envs, device=device)

        state_skill = torch.cat([observations, skill_vec], dim=-1)
        with torch.no_grad():
            actions = agent.get_action(state_skill.to(device), noisy=True).cpu()
        (
            next_observations_raw,
            rewards_raw,
            terminated,
            truncated,
            _,
        ) = environment.step(actions.numpy())

        next_observations = pad_to_dim_2(
            torch.Tensor(next_observations_raw)
        ).to(device)
        rewards = pad_to_dim_2(torch.Tensor(rewards_raw), dim=1).to(device)
        step_reward += reward_scale * rewards.squeeze()

        observations = next_observations
        total_reward += rewards.squeeze()

        if (step + 1) % skill_duration == 0:
            returns.append(step_reward.mean().item())

    if log_writer is not None and update_step is not None:
        log_writer.add_scalar(
            'stats/total_pseudo_reward', total_reward.mean().item(), update_step
        )

    return log_probs, returns, total_reward.mean().item() / episode_length
