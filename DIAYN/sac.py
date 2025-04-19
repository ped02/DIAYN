import copy
import os
import logging
from collections.abc import Mapping
from typing import Type, TypeVar, Any, Optional

import torch
import torch.distributions
from torch.utils.tensorboard import SummaryWriter

from DIAYN import ReplayBuffer

T = TypeVar('T')

logger = None


def setup_logger(verbose=False):
    global logger
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    if not verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)


setup_logger()


class SAC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        policy_class: Type[T],
        q_network_class: Type[T],
        alpha: float,
        policy_kwargs: Mapping[str, Any] = {},
        q_network_kwargs: Mapping[str, Any] = {},
        policy_optimizer_kwargs: Mapping[str, Any] = {},
        q_optimizer_kwargs: Mapping[str, Any] = {},
        policy_gradient_clip_norm: Optional[float] = None,
        q_gradient_clip_norm: Optional[float] = None,
        action_low=None,
        action_high=None,
        log_writer: Optional[SummaryWriter] = None,
        device='cpu',
    ):
        """SAC for continuous control."""
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.device = device
        self.log_writer = log_writer

        self.alpha = alpha
        self.std_dev_low = 0.2
        self.std_dev_high = 2.0
        self.action_low = torch.tensor(
            action_low, device=self.device
        )  # [action_dim]
        self.action_high = torch.tensor(
            action_high, device=self.device
        )  # [acton_dim]

        self.clamp_output = (
            self.action_low is not None and self.action_high is not None
        )
        self.process_action = (
            self.scale_action if self.clamp_output else lambda x: x
        )

        policy_kwargs = policy_kwargs
        q_network_kwargs = q_network_kwargs

        # Instantiate models dynamically with additional parameters
        self.policy = policy_class(state_dim, action_dim, **policy_kwargs).to(
            device
        )
        self.q1 = q_network_class(state_dim, action_dim, **q_network_kwargs).to(
            device
        )
        self.q2 = q_network_class(state_dim, action_dim, **q_network_kwargs).to(
            device
        )

        self.policy_target = copy.deepcopy(self.policy)

        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), **policy_optimizer_kwargs
        )
        self.q_optimizer = torch.optim.Adam(
            [*self.q1.parameters(), *self.q2.parameters()], **q_optimizer_kwargs
        )

        self.policy_gradient_clip_norm = policy_gradient_clip_norm
        self.q_gradient_clip_norm = q_gradient_clip_norm

    def freeze(self):
        def freeze_model(model):
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        freeze_model(self.policy)
        freeze_model(self.q1)
        freeze_model(self.q2)

    def get_state_dict(self, state_dict: Optional[Mapping[str, Any]] = None):
        """
        Return parameters of agent and current weights of models
        """

        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'policy_target_state_dict': self.policy_target.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'alpha': self.alpha,
            'std_dev_low': self.std_dev_low,
            'std_dev_high': self.std_dev_high,
            'action_low': None
            if self.action_low is None
            else self.action_low.cpu().tolist(),
            'action_high': None
            if self.action_high is None
            else self.action_high.cpu().tolist(),
            'clamp_output': self.clamp_output,
            'policy_gradient_clip_norm': self.policy_gradient_clip_norm,
            'q_gradient_clip_norm': self.q_gradient_clip_norm,
        }

        if state_dict is not None:
            checkpoint = state_dict | checkpoint

        return checkpoint

    def set_state_dict(self, state_dict: Mapping[str, Any]):
        """'
        Load parameters of agent and weights of models'
        I guess in case we want to start training from a checkpoint?
        """

        self.policy.load_state_dict(state_dict['policy_state_dict'])
        self.q1.load_state_dict(state_dict['q1_state_dict'])
        self.q2.load_state_dict(state_dict['q2_state_dict'])
        self.policy_target.load_state_dict(
            state_dict['policy_target_state_dict']
        )
        self.q1_target.load_state_dict(state_dict['q1_target_state_dict'])
        self.q2_target.load_state_dict(state_dict['q2_target_state_dict'])

        self.policy_optimizer.load_state_dict(
            state_dict['policy_optimizer_state_dict']
        )
        self.q_optimizer.load_state_dict(state_dict['q_optimizer_state_dict'])

        self.state_dim = state_dict['state_dim']
        self.action_dim = state_dict['action_dim']

        self.alpha = state_dict['alpha']

        self.std_dev_low = state_dict['std_dev_low']
        self.std_dev_high = state_dict['std_dev_high']

        if state_dict.get('action_low') is not None:
            self.action_low = torch.tensor(
                state_dict['action_low'],
                dtype=torch.float32,
                device=self.device,
            )
        if state_dict.get('action_high') is not None:
            self.action_high = torch.tensor(
                state_dict['action_high'],
                dtype=torch.float32,
                device=self.device,
            )

        self.clamp_output = state_dict['clamp_output']
        self.process_action = (
            self.scale_action if self.clamp_output else lambda x: x
        )

        self.policy_gradient_clip_norm = state_dict.get(
            'policy_gradient_clip_norm'
        )

        self.q_gradient_clip_norm = state_dict.get('1_gradient_clip_norm')

    def save_checkpoint(self, filepath: str):
        """
        Save to load using load_checkpoint later
        """

        checkpoint = self.get_state_dict()
        torch.save(checkpoint, filepath)
        logger.info(f'Checkpoint saved to {filepath}')

    def load_checkpoint(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file '{filepath}' not found")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.set_state_dict(checkpoint)

        logger.info(f'Checkpoint loaded from {filepath}')

    def scale_action(self, action):
        """
        Scale action to be between 0 and 1
        """

        z = (torch.nn.functional.tanh(action) + 1.0) / 2.0
        sacled_action = (
            z * (self.action_high - self.action_low) + self.action_low
        )

        return sacled_action

    def get_action_entropy(self, states):
        """
        Get entropy of policy distribution given each state
        """

        mean, log_std_dev = self.policy(states).chunk(2, dim=-1)
        std_dev = log_std_dev.exp().clamp(self.std_dev_low, self.std_dev_high)
        h = torch.distributions.Normal(mean, std_dev).entropy()
        return h

    def get_action(self, states, noisy=False, return_prob=False):
        """
        Sample action from policy network
        """

        mean, log_std_dev = self.policy(states).chunk(2, dim=-1)
        if noisy:
            std_dev = log_std_dev.exp().clamp(
                self.std_dev_low, self.std_dev_high
            )
            dist = torch.distributions.Normal(mean, std_dev)
            action = dist.rsample()
            log_prob_action = dist.log_prob(action)
        else:
            action = mean
            log_prob_action = 0.0

        processed_action = self.process_action(action)
        return (
            processed_action
            if not return_prob
            else (processed_action, log_prob_action)
        )

    def get_action_target(self, states, noisy=False, return_prob=False):
        """Sample action from target policy network"""
        mean, log_std_dev = self.policy_target(states).chunk(2, dim=-1)
        if noisy:
            std_dev = log_std_dev.exp().clamp(
                self.std_dev_low, self.std_dev_high
            )
            dist = torch.distributions.Normal(mean, std_dev)
            action = dist.rsample()
            log_prob_action = dist.log_prob(action)
        else:
            action = mean
            log_prob_action = 0.0

        processed_action = self.process_action(action)
        return (
            processed_action
            if not return_prob
            else (processed_action, log_prob_action)
        )

    def get_q_loss(
        self, states, actions, rewards, next_states, not_dones, gamma=0.99
    ):
        """Calculate SAC q loss"""
        with torch.no_grad():
            policy_action, policy_action_log_prob = self.get_action_target(
                next_states, noisy=True, return_prob=True
            )

            augmented_next_state = torch.concat(
                [next_states, policy_action], dim=-1
            )

            next_q1 = self.q1_target(augmented_next_state)
            next_q2 = self.q2_target(augmented_next_state)

            next_q = torch.minimum(next_q1, next_q2)

        q_target = rewards + gamma * torch.multiply(
            # (next_q+ self.alpha * self.get_action_entropy(next_states).sum(dim=-1)), not_dones
            (next_q - self.alpha * policy_action_log_prob.sum(dim=-1)),
            not_dones,
        )

        augmented_state = torch.concat([states, actions], dim=-1)
        current_q1 = self.q1(augmented_state)
        current_q2 = self.q2(augmented_state)

        q_loss = torch.mean(
            (current_q1 - q_target) ** 2 + (current_q2 - q_target) ** 2
        )

        return q_loss

    def get_policy_loss(self, states, next_states):
        """Calculate SAC policy loss"""
        policy_action, policy_action_log_prob = self.get_action(
            states, noisy=True, return_prob=True
        )

        augmented_state = torch.concat([states, policy_action], dim=-1)
        current_q1 = self.q1(augmented_state)
        current_q2 = self.q2(augmented_state)

        # policy_loss = - torch.mean(torch.minimum(current_q1, current_q2) + self.alpha * self.get_action_entropy(states).sum(dim=-1))
        policy_loss = -torch.mean(
            torch.minimum(current_q1, current_q2)
            - self.alpha * policy_action_log_prob.sum(dim=-1)
        )

        aux_loss = (
            self.policy.get_loss(states, policy_action, next_states)
            if hasattr(self.policy, 'get_loss')
            else torch.tensor(0.0)
        )

        return policy_loss, aux_loss

    def update(
        self,
        replay_buffer: ReplayBuffer,
        step: int,
        q_train_iterations: int,
        policy_train_iterations: int,
        batch_size: int,
        tau: float = 0.1,
    ):
        # Q Train Step
        for _ in range(q_train_iterations):
            q_loss = self.get_q_loss(*replay_buffer.sample(batch_size))
            self.q_optimizer.zero_grad()
            q_loss.backward()
            if self.q_gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [*self.q1.parameters(), *self.q2.parameters()],
                    self.q_gradient_clip_norm,
                )
            self.q_optimizer.step()

        if self.log_writer is not None:
            self.log_writer.add_scalar('loss/q loss', q_loss.item(), step)

        # Policy Train step
        for _ in range(policy_train_iterations):
            states, _, _, next_states, _ = replay_buffer.sample(batch_size)
            policy_loss, policy_aux_loss = self.get_policy_loss(
                states, next_states
            )
            total_policy_loss = policy_loss + policy_aux_loss
            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            if self.policy_gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.policy_gradient_clip_norm
                )
            self.policy_optimizer.step()

        if self.log_writer is not None:
            self.log_writer.add_scalar(
                'loss/-policy loss', -policy_loss.item(), step
            )

            self.log_writer.add_scalar(
                'loss/policy_aux_loss', policy_aux_loss.item(), step
            )

            self.log_writer.add_scalar(
                'loss/total_policy_loss', total_policy_loss.item(), step
            )

        # Action spread logging
        with torch.no_grad():
            _, log_std_dev = self.policy(states).chunk(2, dim=-1)
        std_dev = log_std_dev.exp()
        if self.log_writer is not None:
            self.log_writer.add_scalar(
                'stats/policy std dev', std_dev.mean().item(), step
            )

        # Soft update
        for target_param, param in zip(
            self.q1_target.parameters(), self.q1.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

        for target_param, param in zip(
            self.q2_target.parameters(), self.q2.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

        for target_param, param in zip(
            self.policy_target.parameters(), self.policy.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

        return_dict = {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'action_std_dev': std_dev.mean().item(),
        }

        return return_dict

    def pre_episode(self):
        pass
