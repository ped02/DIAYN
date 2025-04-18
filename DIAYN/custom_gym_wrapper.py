import gymnasium as gym
import numpy as np

from robosuite.wrappers import GymWrapper
import robosuite as suite
from robosuite import load_composite_controller_config


def make_env(config):
    """
    Create a new environment instance.
    Used to create a vector of Gym environments.
    """

    def _thunk():
        print('Creating new environment')

        controller_config = load_composite_controller_config(
            controller=None,
            robot='Panda',
        )

        robosuite_config = {
            'env_name': config['params']['environment_name'],
            'robots': config['params']['robots'],
            'controller_configs': controller_config,
        }

        robosuite_env = suite.make(
            **robosuite_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            render_camera='agentview',
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=False,
        )

        # # Fix cube initial pose
        # robosuite_env.reset()
        # fixed_pos = np.array([0.0, 0.0, 0.85])
        # fixed_quat = np.array([1.0, 0.0, 0.0, 0.0])
        # robosuite_env.sim.data.set_joint_qpos(
        #     'cube:joint0',
        #     np.concatenate([fixed_pos, fixed_quat])
        # )
        # robosuite_env.sim.forward()

        env = CustomGymWrapper(robosuite_env, config)

        # Ensure metadata exists and is a dict before modifying
        if env.metadata is None:
            env.metadata = {}
        env.metadata['render_modes'] = []
        env.metadata['autoreset'] = False

        return env

    return _thunk


class CustomGymWrapper(gym.ObservationWrapper):
    def __init__(self, robosuite_env, config):
        # Wrap robosuite with GymWrapper inside
        gym_env = GymWrapper(robosuite_env)
        super().__init__(gym_env)

        # Flags for which state information to include in the observation space
        self.use_eef_state = config['observations']['use_eef_state']
        self.use_joint_vels = config['observations']['use_joint_vels']
        self.use_cube_pos = config['observations']['use_cube_pos']

        # Set the observation space
        observation_raw = self.get_observation_raw()

        # Update observation space specification based on new state information
        new_dim = observation_raw.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_dim,),
            dtype=np.float32,
        )

    def get_observation_raw(self):
        """
        Convert obs_dict outputted by robosuite into an array.
        Only adds state information specified by ur5e_config.yaml file
        """

        obs_dict = self.env.unwrapped._get_observations()
        # print(obs_dict)
        # exit()

        observation_raw = np.concatenate(
            [
                *(
                    [obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat']]
                    if self.use_eef_state
                    else []
                ),
                *(
                    [obs_dict['robot0_joint_vel']]
                    if self.use_joint_vels
                    else []
                ),
                *([obs_dict['cube_pos']] if self.use_cube_pos else []),
            ]
        )

        return observation_raw

    def observation(self, obs):
        observation_raw = self.get_observation_raw()

        return observation_raw
