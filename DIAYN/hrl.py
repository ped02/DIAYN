import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

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


class HighLevelPolicy(nn.Module):
    """
    Hierarchical Reinforcement Learning: High-Level Controller
    """

    def __init__(
        self,
        state_dim: int,
        skill_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        log_writer: Optional[SummaryWriter] = None,
        device: str = 'cpu',
    ):
        super().__init__()

        self.device = device
        self.log_writer = log_writer

        # Predict skill logits
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, skill_dim),
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.skill_dim = skill_dim
        self.state_dim = state_dim

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)
        logits = self.policy_net(state)
        return logits

    def select_skill(self, state, deterministic=False):
        logits = self.forward(state)
        distribution = torch.distributions.Categorical(logits=logits)

        if deterministic:
            skill = torch.argmax(logits, dim=-1)
        else:
            skill = distribution.sample()

        log_prob = distribution.log_prob(skill)

        return skill.item(), log_prob

    def update(self, log_probs, returns, step: int):
        """
        Update policy using REINFORCE
        Args:
            log_probs: list of log_probs of selected skills
            returns: list of corresponding returns (sum of pseudo-rewards)
            step: current training step
        """
        log_probs = torch.stack(log_probs).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.log_writer is not None:
            self.log_writer.add_scalar(
                'loss/high_level_policy', loss.item(), step
            )

        logger.info(f'[Step {step}] High-Level Policy Loss: {loss.item():.4f}')

        return {'high_level_policy_loss': loss.item()}

    def save_checkpoint(self, filepath: str):
        checkpoint = {
            'policy_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'skill_dim': self.skill_dim,
            'state_dim': self.state_dim,
        }
        torch.save(checkpoint, filepath)
        logger.info(f'High-Level Policy checkpoint saved to {filepath}')

    def load_checkpoint(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file '{filepath}' not found")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.skill_dim = checkpoint['skill_dim']
        self.state_dim = checkpoint['state_dim']

        logger.info(f'High-Level Policy checkpoint loaded from {filepath}')

    def pre_episode(self):
        pass
