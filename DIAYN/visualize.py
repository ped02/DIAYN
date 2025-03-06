import torch

import gymnasium as gym

from DIAYN import AgentBase

def visualize(
        environment_name: str,
        max_episode_steps: int,
        agent: AgentBase,
        device,
        output_folder: str = './',
        output_name_prefix: str = 'rl-video'
        ):
    """Visualize Agent Interaction. Run until episode ends or max_episode_step is reached.
    
    Parameters
    ----------
    environment_name: str
        name of gym environment
    max_episode_steps: int
        number of steps in an episode to record for
    agent: AgentBase
        Agent to visualize
    device: str or pytorch device
        Device of agent
    output_folder: str
        Path to directory of output
    output_name_prefix: str
        Name prefix of output file
    """

    env = gym.make(environment_name, render_mode='rgb_array')

    env_recorder = gym.wrappers.RecordVideo(env, video_folder=output_folder, name_prefix=output_name_prefix, episode_trigger=lambda x:True)

    observations_raw, _ = env_recorder.reset()

    for t in range(max_episode_steps):
        with torch.no_grad():
            actions = agent.get_action(torch.Tensor(observations_raw).to(device)[None, :]).squeeze(0)
        observations_raw, rewards, done, _, _ = env_recorder.step(actions.cpu().numpy())
        if done:
            break

    env_recorder.close()
