import os
from typing import Optional, Union
import yaml
import torch
from pathlib import Path

from DIAYN import make_env, visualize_robosuite
from DIAYN.hrl import HighLevelPolicy


from gymnasium.vector import SyncVectorEnv


def main(
    environment_name: str,
    robots: str,
    steps_per_episode: int,
    num_skills: int,
    visualize_skill: Union[int, None],
    video_output_folder: str,
    video_file_prefix: str = 'rl_video',
    model_load_path: Optional[str] = None,
    evaluate_episodes: int = 10,
    config: Optional[dict] = None,
):
    device = torch.device('cuda')

    envs = SyncVectorEnv(
        [make_env(config) for _ in range(config['params']['num_envs'])]
    )
    observation_dims = envs.observation_space.shape[1]
    # action_dims = envs.action_space.shape[1]

    # action_low = envs.action_space.low[0]
    # action_high = envs.action_space.high[0]

    # Load High-Level Policy
    high_level_policy = HighLevelPolicy(
        state_dim=observation_dims,
        skill_dim=num_skills,
        device=device,
        log_writer=None,
    )

    if model_load_path is not None:
        high_level_policy.load_checkpoint(model_load_path)

    if visualize_skill is None:
        for z in range(num_skills):
            print(f'Visualizing Skill {z}')
            visualize_robosuite(
                environment_name,
                robots,
                steps_per_episode,
                agent=None,
                device=device,
                skill_index=z,
                num_skills=num_skills,
                output_folder=video_output_folder,
                output_name_prefix=f'{video_file_prefix}_skill{z}',
                config=config,
            )
    else:
        visualize_robosuite(
            environment_name,
            robots,
            steps_per_episode,
            agent=None,
            device=device,
            skill_index=visualize_skill,
            num_skills=num_skills,
            output_folder=video_output_folder,
            output_name_prefix=f'{video_file_prefix}_skill{visualize_skill}',
            config=config,
        )


def checkpoint_check(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file '{filepath}' not found")

    checkpoint = torch.load(filepath, map_location='cuda')
    print('Checkpoint contents:', checkpoint)


if __name__ == '__main__':
    current_dir = str(Path(__file__).parent.resolve())
    print('Current directory:', current_dir)

    with open(current_dir + '/config/ur5e_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    environment_name = config['params']['environment_name']
    robots = config['params']['robots']
    num_steps = config['evaluation_params']['num_steps']
    num_skills = config['params']['num_skills']

    model_load_path = config['hrl_file_params']['model_load_path']
    video_output_folder = config['hrl_file_params']['video_output_folder']
    video_prefix_path = config['hrl_file_params']['video_prefix_path']

    visualize_skill = None

    if video_output_folder is not None:
        os.makedirs(video_output_folder, exist_ok=True)

    main(
        environment_name,
        robots,
        num_steps,
        num_skills,
        visualize_skill,
        video_output_folder,
        video_prefix_path,
        model_load_path=model_load_path,
        config=config,
    )
