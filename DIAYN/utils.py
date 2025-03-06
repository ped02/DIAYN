import torch

def replay_post_processor(samples, device):
    return [ torch.stack(e).to(device) for e in zip(*samples) ]

def pad_to_dim_2(tensor: torch.Tensor):
    """Accepts 1-2 dim tensor. Return 2 dim"""
    return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor

def augment_state_with_skill(state: torch.Tensor, skill: int, skill_dim: int):
    """Augments the state tensor by concatenating a one-hot encoded skill vector.
    
    Args:
        state (torch.Tensor): Tensor of shape [N, obs_dim].
        skill (int): Skill index to encode.
        skill_dim (int): Number of possible skills (dimension of one-hot vector).
    
    Returns:
        torch.Tensor: Augmented state tensor of shape [N, obs_dim + skill_dim].
    """
    skill_one_hot = torch.nn.functional.one_hot(
        torch.tensor(skill, device=state.device), num_classes=skill_dim
    ).to(state.dtype)

    skill_one_hot = skill_one_hot.expand(state.shape[0], -1)

    return torch.cat([state, skill_one_hot], dim=-1)
