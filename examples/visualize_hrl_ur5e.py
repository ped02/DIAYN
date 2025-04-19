import os
import torch
import yaml
from pathlib import Path
from typing import Optional, Union

from DIAYN import make_env, visualize_robosuite, DIAYNAgent
from DIAYN.hrl import HighLevelPolicy
from gymnasium.vector import SyncVectorEnv


def create_hrl_agent(high_level_policy, skill_policy, num_skills, device):
    class HRLWrapper:
        def __init__(self, high_level_policy, skill_policy):
            self.high_level_policy = high_level_policy
            self.skill_policy = skill_policy
            self.device = device
            self.num_skills = num_skills

        def get_action(self, obs, noisy=True):
            logits = self.high_level_policy(obs.to(self.device))
            skill_distribution = torch.distributions.Categorical(logits=logits)
            sampled_skill = skill_distribution.sample()
            skill_onehot = torch.nn.functional.one_hot(
                sampled_skill, num_classes=self.num_skills
            ).float()

            # DIAYN expects state + skill input
            state_skill = torch.cat(
                [obs.to(self.device), skill_onehot.to(self.device)], dim=-1
            )
            action = self.skill_policy.get_action(state_skill, noisy=noisy)
            return action

    return HRLWrapper(high_level_policy, skill_policy)


def load_skill_policy(
    model_path, obs_dim, act_dim, num_skills, action_low, action_high, device
):
    def get_q_network(observation_dim, action_dim):
        return torch.nn.Sequential(
            torch.nn.Linear(observation_dim + action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def get_policy_network(observation_dim, action_dim):
        return torch.nn.Sequential(
            torch.nn.Linear(observation_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2 * action_dim),
        )

    def get_discriminator(observation_dim, skill_dim):
        return torch.nn.Sequential(
            torch.nn.LayerNorm(observation_dim),
            torch.nn.Linear(observation_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, skill_dim),
        )

    policy = DIAYNAgent(
        skill_dim=num_skills,
        discriminator_class=get_discriminator,
        discriminator_optimizer_kwargs={'lr': 4e-4},
        state_dim=obs_dim,
        action_dim=act_dim,
        policy_class=get_policy_network,
        q_network_class=get_q_network,
        alpha=0.1,
        action_low=action_low,
        action_high=action_high,
        policy_optimizer_kwargs={'lr': 3e-5},
        q_optimizer_kwargs={'lr': 1e-4},
        device=device,
    )
    policy.load_checkpoint(model_path)
    policy.policy.eval()
    for p in policy.policy.parameters():
        p.requires_grad = False
    return policy


def main(
    environment_name: str,
    robots: str,
    steps_per_episode: int,
    num_skills: int,
    visualize_skill: Union[int, None],
    video_output_folder: str,
    video_file_prefix: str = 'rl_video',
    model_load_path: Optional[str] = None,
    config: Optional[dict] = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    envs = SyncVectorEnv(
        [make_env(config) for _ in range(config['params']['num_envs'])]
    )
    observation_dims = envs.observation_space.shape[1]
    action_dims = envs.action_space.shape[1]
    action_low = envs.action_space.low[0]
    action_high = envs.action_space.high[0]

    high_level_policy = HighLevelPolicy(
        state_dim=observation_dims,
        skill_dim=num_skills,
        device=device,
        log_writer=None,
    )

    if model_load_path is not None:
        print(f'Loading high-level policy from {model_load_path}')
        high_level_policy.load_checkpoint(model_load_path)

    skill_model_path = config['hrl_training_params'][
        'model_load_path_pretrained_diayn'
    ]
    skill_policy = load_skill_policy(
        skill_model_path,
        obs_dim=observation_dims,
        act_dim=action_dims,
        num_skills=num_skills,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )

    hrl_agent = create_hrl_agent(
        high_level_policy, skill_policy, num_skills, device
    )

    if visualize_skill is None:
        for z in range(num_skills):
            print(f'Visualizing Skill {z}')
            visualize_robosuite(
                environment_name,
                robots,
                steps_per_episode,
                agent=hrl_agent,
                device=device,
                skill_index=None,
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
            agent=hrl_agent,
            device=device,
            skill_index=visualize_skill,
            num_skills=num_skills,
            output_folder=video_output_folder,
            output_name_prefix=f'{video_file_prefix}_skill{visualize_skill}',
            config=config,
        )


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
