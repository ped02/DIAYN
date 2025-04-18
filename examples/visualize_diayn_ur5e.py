import os
from typing import Optional, Union
import yaml
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

from DIAYN import DIAYNAgent, CustomGymWrapper, make_env, evaluate_agent_robosuite, visualize_robosuite

import robosuite as suite
from robosuite import load_composite_controller_config

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

    # Setup logging
    log_writer = None

    # Setup env variables

    # TODO: Just get dims directly instead of dealing with this each time
    envs = SyncVectorEnv(
        [make_env(config) for _ in range(num_envs)]
    )
    observation_dims = envs.observation_space.shape[1]
    action_dims = envs.action_space.shape[1]

    action_low = envs.action_space.low[0]
    action_high = envs.action_space.high[0]

    # Setup networks
    def get_q_network(observation_dim, action_dim):
        q_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim + action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

        return q_network

    def get_policy_network(observation_dim, action_dim):
        policy_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2 * action_dim),
        )
        return policy_network

    def get_discriminiator_network(observation_dim, skill_dim):
        # Output logits
        discriminiator_network = torch.nn.Sequential(
            torch.nn.LayerNorm(observation_dim),
            torch.nn.Linear(observation_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 256),
            # torch.nn.ReLU(),
            torch.nn.Linear(128, skill_dim),
        )

        return discriminiator_network

    q_optimizer_kwargs = {'lr': 1e-3}
    discriminator_optimizer_kwargs = {'lr': 4e-4}
    policy_optimizer_kwargs = {'lr': 1e-4}

    # Setup agent
    diayn_agent = DIAYNAgent(
        skill_dim=num_skills,
        discriminator_class=get_discriminiator_network,
        discriminator_optimizer_kwargs=discriminator_optimizer_kwargs,
        state_dim=observation_dims,
        action_dim=action_dims,
        policy_class=get_policy_network,
        q_network_class=get_q_network,
        alpha=0.1,
        action_low=action_low,
        action_high=action_high,
        policy_optimizer_kwargs=policy_optimizer_kwargs,
        q_optimizer_kwargs=q_optimizer_kwargs,
        device=device,
        log_writer=log_writer,
    )

    if model_load_path is not None:
        diayn_agent.load_checkpoint(model_load_path)

    # Skill
    if visualize_skill is None:
        skills_mean_step_reward = []
        skills_return = []
        for z in range(num_skills):
            visualize_robosuite(
                environment_name,
                robots,
                steps_per_episode,
                diayn_agent,
                device,
                skill_index=z,
                num_skills=num_skills,
                output_folder=video_output_folder,
                output_name_prefix=f'{video_file_prefix}_skill{z}',
                config=config,
            )

            # result_dict = evaluate_agent_robosuite(
            #     environment_name,
            #     robots,
            #     steps_per_episode,
            #     evaluate_episodes,
            #     diayn_agent,
            #     device,
            #     skill_index=z,
            #     num_skills=num_skills,
            #     config= config,
            # )
            # skills_return.append(np.mean(result_dict['total_return']))
            # skills_mean_step_reward.append(
            #     np.nanmean(
            #         result_dict['total_return'] / result_dict['episode_length']
            #     )
            # )
            # print(
            #     f'[Skill {z}] Mean Step Reward: {skills_mean_step_reward[-1]} Mean Total Return: {skills_return[-1]}'
            # )

        # with open(
        #     os.path.join(video_output_folder, 'skills_order_step_reward'), 'w'
        # ) as f:
        #     f.write(
        #         '\n'.join(
        #             [
        #                 f'{skill}: {total_return}'
        #                 for skill, total_return in sorted(
        #                     enumerate(skills_mean_step_reward),
        #                     key=lambda x: x[1],
        #                     reverse=True,
        #                 )
        #             ]
        #         )
        #     )

        with open(
            os.path.join(video_output_folder, 'skills_order_total_return'), 'w'
        ) as f:
            f.write(
                '\n'.join(
                    [
                        f'{skill}: {total_return}'
                        for skill, total_return in sorted(
                            enumerate(skills_return),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    ]
                )
            )

    else:
        visualize_robosuite(
            environment_name,
            robots,
            steps_per_episode,
            diayn_agent,
            device,
            skill_index=visualize_skill,
            num_skills=num_skills,
            output_folder=video_output_folder,
            output_name_prefix=f'{video_file_prefix}_skill{visualize_skill}',
        )
        # result_dict = evaluate_agent_robosuite(
        #     environment_name,
        #     robots,
        #     steps_per_episode,
        #     evaluate_episodes,
        #     diayn_agent,
        #     device,
        #     skill_index=visualize_skill,
        #     num_skills=num_skills,
        #     config=config,
        # )
        # print(
        #     f'[Skill {visualize_skill}] Mean Step Reward: {np.nanmean(result_dict["total_return"]/result_dict["episode_length"])} Mean Total Return: {np.mean(result_dict["total_return"])}'
        # )


def checkpoint_check(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file '{filepath}' not found")

    checkpoint = torch.load(filepath, map_location='cuda')
    print('Checkpoint contents:', checkpoint)  # Add this line to debug
    # self.set_state_dict(checkpoint)

    # logger.info(f'Checkpoint loaded from {filepath}')


if __name__ == '__main__':
    # Read config file for all settings
    current_dir = str(Path(__file__).parent.resolve())
    print('Current directory:', current_dir)

    with open(current_dir + '/config/ur5e_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # exit()
    environment_name = config['params']['environment_name']
    robots = config['params']['robots']
    num_envs = config['params']['num_envs']
    num_steps = config['evaluation_params']['num_steps']
    num_skills = config['params']['num_skills']

    # Folder paths
    model_load_path = config['file_params']['model_load_path']
    video_output_folder = config['file_params']['video_output_folder']
    video_prefix_path = config['file_params']['video_prefix_path']

    # If we want a specific skill to be visualized
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
        config=config
    )
