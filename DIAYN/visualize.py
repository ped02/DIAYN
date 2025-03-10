from typing import Optional

import torch

import gymnasium as gym

from DIAYN import AgentBase
from DIAYN.utils import pad_to_dim_2


def visualize(
    environment_name: str,
    max_episode_steps: int,
    agent: AgentBase,
    device,
    skill_index: Optional[int] = None,
    num_skills: Optional[int] = None,
    output_folder: str = './',
    output_name_prefix: str = 'rl-video',
):
    """Visualize Agent Interaction. Run until episode ends or max_episode_step is reached.

    Args:
        environment_name (str): name of gym environment
        max_episode_steps (int): number of steps in an episode to record for
        agent (AgentBase): Agent to visualize
        device (str or pytorch device): Device of agent
        output_folder (str): Path to directory of output
        output_name_prefix (str): Name prefix of output file
    """

    skill_vector = None
    if skill_index is not None:
        skill_vector = torch.nn.functional.one_hot(
            torch.tensor(skill_index, device=device), num_classes=num_skills
        ).unsqueeze(0)

    def augment_state_skill(observation_raw):
        return torch.cat(
            [
                pad_to_dim_2(torch.Tensor(observation_raw).to(device)),
                skill_vector,
            ],
            dim=-1,
        )

    env = gym.make(environment_name, render_mode='rgb_array')

    def process_pure_state(observation_raw):
        return pad_to_dim_2(torch.Tensor(observation_raw).to(device))

    process_observation = process_pure_state
    if skill_index is not None:
        process_observation = augment_state_skill

    env_recorder = gym.wrappers.RecordVideo(
        env,
        video_folder=output_folder,
        name_prefix=output_name_prefix,
        episode_trigger=lambda x: True,
    )

    observations_raw, _ = env_recorder.reset()
    observation = process_observation(observations_raw)

    for t in range(max_episode_steps):
        with torch.no_grad():
            actions = agent.get_action(observation).squeeze(0)
        observations_raw, rewards, done, _, _ = env_recorder.step(
            actions.cpu().numpy()
        )

        observation = process_observation(observations_raw)

        if done:
            break

    env_recorder.close()
