import os
import torch
import yaml
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import datetime

from DIAYN import DIAYNAgent, make_env
from DIAYN.hrl import HighLevelPolicy
from DIAYN.rl_rollout import rollout_hrl
from gymnasium.vector import SyncVectorEnv


def one_hot(index, dim):
    return torch.nn.functional.one_hot(
        torch.tensor(index), num_classes=dim
    ).float()


def main():
    # === Load Config ===
    config_path = Path(__file__).parent / 'config' / 'ur5e_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_envs = config['params']['num_envs']
    num_skills = config['params']['num_skills']
    episodes = config['hrl_training_params']['episodes']
    episode_length = config['hrl_training_params'].get('num_steps', 1000)
    skill_duration = config['hrl_training_params'].get('skill_duration', 50)
    model_load_path_diayn = config['hrl_training_params'][
        'model_load_path_pretrained_diayn'
    ]
    model_load_path_hrl = config['hrl_training_params'].get(
        'model_load_path', None
    )

    run_name = 'run_hrl_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S'
    )
    log_path = os.path.join(
        config['hrl_training_params']['log_parent_folder'], run_name
    )
    model_save_folder = os.path.join(
        config['hrl_training_params']['model_save_folder'], run_name
    )
    os.makedirs(model_save_folder, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    training_state_path = os.path.join(model_save_folder, 'training_state.pt')

    # === Load Env ===
    envs = SyncVectorEnv([make_env(config) for _ in range(num_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Load DIAYN agent and freeze ===
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

    def get_discriminiator_network(observation_dim, skill_dim):
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

    q_optimizer_kwargs = {'lr': 1e-4}
    discriminator_optimizer_kwargs = {'lr': 4e-4}
    policy_optimizer_kwargs = {'lr': 3e-5}

    action_low = envs.action_space.low[0]
    action_high = envs.action_space.high[0]

    diayn_agent = DIAYNAgent(
        skill_dim=num_skills,
        discriminator_class=get_discriminiator_network,
        discriminator_optimizer_kwargs=discriminator_optimizer_kwargs,
        state_dim=obs_dim,
        action_dim=action_dim,
        policy_class=get_policy_network,
        q_network_class=get_q_network,
        alpha=0.1,
        action_low=action_low,
        action_high=action_high,
        policy_optimizer_kwargs=policy_optimizer_kwargs,
        q_optimizer_kwargs=q_optimizer_kwargs,
        device=device,
    )
    diayn_agent.load_checkpoint(model_load_path_diayn)
    diayn_agent.policy.eval()
    for param in diayn_agent.policy.parameters():
        param.requires_grad = False

    # === Init High-Level Policy ===
    high_level = HighLevelPolicy(
        state_dim=obs_dim,
        skill_dim=num_skills,
        device=device,
        log_writer=writer,
    )

    if model_load_path_hrl is not None and os.path.exists(model_load_path_hrl):
        print(
            f'Loading high-level policy checkpoint from {model_load_path_hrl}'
        )
        high_level.load_checkpoint(model_load_path_hrl)

    update_step = 0
    if os.path.exists(training_state_path):
        print(f'Resuming training from {training_state_path}')
        checkpoint = torch.load(training_state_path, map_location=device)
        update_step = checkpoint.get('step', 0)

    for ep in range(episodes):
        log_probs, returns, total_reward = rollout_hrl(
            environment=envs,
            episode_length=episode_length,
            skill_duration=skill_duration,
            agent=diayn_agent,
            high_level_policy=high_level,
            device=device,
            num_skills=num_skills,
            reward_scale=1.0,
            log_writer=writer,
            update_step=update_step,
        )

        high_level.update(log_probs, returns, update_step)
        update_step += 1

        if (ep + 1) % 10 == 0:
            ckpt_path = os.path.join(
                model_save_folder, f'checkpoint_ep{ep+1}.pt'
            )
            print(f'Saving checkpoint at episode {ep+1} to {ckpt_path}')
            high_level.save_checkpoint(ckpt_path)
            print(f'Episode {ep+1} completed')

    print('Training complete.')
    final_path = os.path.join(model_save_folder, 'final_high_level_policy.pt')
    print(f'Saving final high-level policy to {final_path}')
    high_level.save_checkpoint(final_path)
    torch.save({'step': update_step}, training_state_path)
    writer.close()


if __name__ == '__main__':
    main()
