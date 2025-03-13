import os
from typing import Optional, Union

import numpy as np

import torch

import gymnasium as gym

from DIAYN import DIAYNAgent, evaluate_agent, visualize

DIAYN_PARAMS = {
    'layer_size': 300,
    'batch_size': 128,
    'num_steps': 1000,
    'num_epochs': 10000,
    'lr': 3e-4,
}

def main(
    environment_name: str,
    steps_per_episode: int,
    num_skills: int,
    visualize_skill: Union[int, None],
    video_output_folder: str,
    video_file_prefix: str = 'rl_video',
    model_load_path: Optional[str] = None,
    evaluate_episodes: int = 10,
):
    device = torch.device('cuda')

    # Setup logging
    log_writer = None

    # Setup env variables
    envs = gym.make_vec(
        environment_name,
        vectorization_mode=gym.VectorizeMode.SYNC,
        num_envs=1,
    )

    observation_dims = envs.observation_space.shape[1]
    action_dims = envs.action_space.shape[1]

    action_low = envs.action_space.low[0]
    action_high = envs.action_space.high[0]

    # Setup networks
    def get_q_network(observation_dim, action_dim):
        layer_size = DIAYN_PARAMS['layer_size']
        q_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim + action_dim, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, 1),
        )

        return q_network

    def get_policy_network(observation_dim, action_dim):
        layer_size = DIAYN_PARAMS['layer_size']
        policy_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, 2 * action_dim),
        )
        return policy_network

    def get_discriminiator_network(observation_dim, skill_dim):
        # Output logits
        layer_size = DIAYN_PARAMS['layer_size']
        discriminiator_network = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, skill_dim),
        )

        return discriminiator_network

    q_optimizer_kwargs = {'lr': DIAYN_PARAMS['lr']}
    discriminator_optimizer_kwargs = {'lr': 4e-4}
    policy_optimizer_kwargs = {'lr': DIAYN_PARAMS['lr']}

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
            visualize(
                environment_name,
                steps_per_episode,
                diayn_agent,
                device,
                skill_index=z,
                num_skills=num_skills,
                output_folder=video_output_folder,
                output_name_prefix=f'{video_file_prefix}_skill{z}',
            )

            result_dict = evaluate_agent(
                environment_name,
                steps_per_episode,
                evaluate_episodes,
                diayn_agent,
                device,
                skill_index=z,
                num_skills=num_skills,
            )
            skills_return.append(np.mean(result_dict['total_return']))
            skills_mean_step_reward.append(
                np.nanmean(
                    result_dict['total_return'] / result_dict['episode_length']
                )
            )
            print(
                f'[Skill {z}] Mean Step Reward: {skills_mean_step_reward[-1]} Mean Total Return: {skills_return[-1]}'
            )

        with open(
            os.path.join(video_output_folder, 'skills_order_step_reward'), 'w'
        ) as f:
            f.write(
                '\n'.join(
                    [
                        f'{skill}: {total_return}'
                        for skill, total_return in sorted(
                            enumerate(skills_mean_step_reward),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    ]
                )
            )

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
        visualize(
            environment_name,
            steps_per_episode,
            diayn_agent,
            device,
            skill_index=visualize_skill,
            num_skills=num_skills,
            output_folder=video_output_folder,
            output_name_prefix=f'{video_file_prefix}_skill{visualize_skill}',
        )
        result_dict = evaluate_agent(
            environment_name,
            steps_per_episode,
            evaluate_episodes,
            diayn_agent,
            device,
            skill_index=visualize_skill,
            num_skills=num_skills,
        )
        print(
            f'[Skill {visualize_skill}] Mean Step Reward: {np.nanmean(result_dict["total_return"]/result_dict["episode_length"])} Mean Total Return: {np.mean(result_dict["total_return"])}'
        )

def checkpoint_check(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file '{filepath}' not found")

    checkpoint = torch.load(filepath, map_location='cuda')
    print("Checkpoint contents:", checkpoint)  # Add this line to debug
    # self.set_state_dict(checkpoint)

    # logger.info(f'Checkpoint loaded from {filepath}')

if __name__ == '__main__':
    environment_name = 'Hopper-v5'

    episodes = 1000
    num_envs = 4
    num_steps = 1000 # 1000
    num_skills = 50
    
    checkpoint_name = 'checkpoint_600'
    model_load_path = '/home/rmineyev3/DIAYN/examples/hopper_runs/run_6/weights/' + checkpoint_name + ".pt"

    # checkpoint_check(model_load_path)

    visualize_skill = None



    # Check if output folder exists. If not, create it
    video_output_folder = '/home/rmineyev3/DIAYN/examples/hopper_runs/run_6/weights/' + checkpoint_name + '_videos'
    if video_output_folder is not None:
        os.makedirs(video_output_folder, exist_ok=True)

    video_prefix_path = 'rl_video'

    main(
        environment_name,
        num_steps,
        num_skills,
        visualize_skill,
        video_output_folder,
        video_prefix_path,
        model_load_path=model_load_path,
    )
