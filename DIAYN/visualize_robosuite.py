from typing import Optional

import torch

import gymnasium as gym

from DIAYN import AgentBase
from DIAYN.utils import pad_to_dim_2

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

import robosuite as suite
from robosuite.wrappers import GymWrapper

import numpy as np
import imageio
import os

# def make_env(env_name, robots):
#     def _thunk():
#         print("Creating new environment")

#         controller_config = load_composite_controller_config(
#             controller=None,
#             robot="Panda",
#         )

#         config = {
#             "env_name": env_name,
#             "robots": robots,
#             "controller_configs": controller_config,
#         }

#         robosuite_env = suite.make(
#             **config,
#             has_renderer=True,
#             has_offscreen_renderer=True,
#             render_camera="agentview",
#             ignore_done=True,
#             use_camera_obs=False,
#             reward_shaping=True,
#             control_freq=20,
#             hard_reset=False,
#         )

#         env = GymWrapper(robosuite_env)

#         # ✅ Patch the render method for Gymnasium compatibility
#         def _render():
#             return robosuite_env.render()
        
#         env.render = _render

#         # Ensure metadata exists and supports video recording
#         env.metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
#         env.render_mode = "rgb_array"

#         return env

#     return _thunk

def visualize_robosuite(
    environment_name: str,
    robots: str,
    max_episode_steps: int,
    agent: AgentBase,
    device,
    skill_index: Optional[int] = None,
    num_skills: Optional[int] = None,
    output_folder: str = './',
    output_name_prefix: str = 'rl-video',
    config: Optional[dict] = None,
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

    # env = gym.make(environment_name, render_mode='rgb_array')
    # env = make_env(environment_name, robots)()

    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots,
    )

    robosuite_config = {
        "env_name": environment_name,
        "robots": robots,
        "controller_configs": controller_config,
    }

    camera_view = config['evaluation_params']['camera_view']

    env = suite.make(
            **robosuite_config,
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera=camera_view,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=False,
        )
    
    video_height = 512
    video_width = 512

    def process_pure_state(observation_raw):
        return pad_to_dim_2(torch.Tensor(observation_raw).to(device))
    
    def convert_obs_dict_to_nparray(obs_dict, config):
        use_eef_state = config['observations']['use_eef_state']
        use_joint_vels = config['observations']['use_joint_vels']
        use_cube_pos = config['observations']['use_cube_pos']

        observation_raw = np.concatenate([
                    *([obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat']] if use_eef_state else []),
                    *( [obs_dict['robot0_joint_vel']] if use_joint_vels else []),
                    *( [obs_dict['cube_pos']] if use_cube_pos else []),
                ])
        
        return observation_raw

    process_observation = process_pure_state
    if skill_index is not None:
        process_observation = augment_state_skill

    # env_recorder = gym.wrappers.RecordVideo(
    #     env,
    #     video_folder=output_folder,
    #     name_prefix=output_name_prefix,
    #     episode_trigger=lambda x: True,
    # )



    obs_dict = env.reset()

    observation_raw = convert_obs_dict_to_nparray(obs_dict, config)
    observation = process_observation(observation_raw)

    frames = []

    for t in range(max_episode_steps):
        with torch.no_grad():
            actions = agent.get_action(observation).squeeze(0)
        obs_dict, rewards, done, _ = env.step(
            actions.cpu().numpy()
        )

        observations_raw = convert_obs_dict_to_nparray(obs_dict, config)

        observation = process_observation(observations_raw)

        # Render frame
        frame = env.sim.render(
            camera_name=camera_view,
            height=video_height,
            width=video_width,
            depth=False,
        )

        frame = np.flipud(frame).astype(np.uint8)
        frames.append(frame)

        if done:
            break
    
    # Close video window
    env.close()

    # Save video
    video_path = os.path.join(output_folder, f"{output_name_prefix}.mp4")
    print(f"Saving video to {video_path}")
    imageio.mimsave(video_path, frames, fps=20)