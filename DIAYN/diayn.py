from collections.abc import Mapping
from typing import Type, TypeVar, Any, Union, Optional

import torch

from DIAYN import SAC

T = TypeVar('T')

class DIAYNAgent(SAC):

    def __init__(
            self,
            skill_dim: int,
            discriminator_class: Type[T],
            discriminator_optimizer_kwargs: Mapping[str, Any] = {},
            **kwargs):
        
        state_dim = kwargs['state_dim']
        # Modify before constructing the Q and Policy 
        kwargs['state_dim'] += skill_dim
        super().__init__(**kwargs)

        # Modify Back
        self.state_dim = state_dim

        self.skill_dim = skill_dim

        # Setup discriminator
        self.discriminator = discriminator_class(self.state_dim, self.skill_dim).to(self.device)

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), **discriminator_optimizer_kwargs)

    def update(self, replay_buffer, step, disciminator_iterations, q_train_iterations, policy_train_iterations, batch_size, tau = 0.1):

        # Q Train Step
        for _ in range(q_train_iterations):
            state_skill, skill_index, actions, _, next_state_skill, not_dones = replay_buffer.sample(batch_size)

            state, skill = torch.split(state_skill, [self.state_dim, self.skill_dim], dim=-1)

            with torch.no_grad():
                discriminator_logits = self.discriminator(state)

            rewards = torch.nn.functional.log_softmax(discriminator_logits, dim=-1).gather(-1, skill_index) + torch.log(torch.tensor(self.skill_dim, device=self.device)) # Assume uniform distribution

            q_loss = self.get_q_loss(state_skill, actions, rewards, next_state_skill, not_dones)
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

        if self.log_writer is not None:
            self.log_writer.add_scalar('loss/q loss', q_loss.item(), step)

        # Policy Train step
        for _ in range(policy_train_iterations):
            state_skill, _, _, _, _, _ = replay_buffer.sample(batch_size)
            policy_loss = self.get_policy_loss(state_skill)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        if self.log_writer is not None:
            self.log_writer.add_scalar('loss/-policy loss', - policy_loss.item(), step)

        # Discriminator Train Step
        for _ in range(disciminator_iterations):
            state_skill, skill_index, _, _, _, _ = replay_buffer.sample(batch_size)
            state, skill = torch.split(state_skill, [self.state_dim, self.skill_dim], dim=-1)

            discriminator_logits = self.discriminator(state)

            discriminator_loss = torch.nn.functional.cross_entropy(discriminator_logits, skill_index.squeeze(-1))
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

        if self.log_writer is not None:
            self.log_writer.add_scalar('loss/discriminator loss', discriminator_loss.item(), step)

        # Action spread logging
        with torch.no_grad():
            _, log_std_dev = self.policy(state_skill).chunk(2,  dim=-1)
        std_dev = log_std_dev.exp()
        if self.log_writer is not None:
            self.log_writer.add_scalar('stats/policy std dev', std_dev.mean().item(), step)

        # Soft update
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        return_dict = {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'action_std_dev': std_dev.mean().item()
        }

        return return_dict