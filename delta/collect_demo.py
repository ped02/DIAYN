"""Teleoperate robot with keyboard.

Default environment is Lift with a UR5e arm. Demonstration data is saved as a list of tuples, with each tuple having (obs, act).
Demonstrations are saved when pressing CTRL+Q (reset) in the visualization window or CTRL+C (KeyboardInterrupt) in terminal.

To run:
    (mac)   $ mjpython collect_demo.py --robots <OPTIONAL> --environment <OPTIONAL>
    (linux) $ python collect_demo.py --robots <OPTIONAL> --environment <OPTIONAL>
If --robots and --environment are omitted, defaults are used.
"""

import argparse
import time

import numpy as np

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler

import pickle  # nosec
import glob
import os

# Setup saving trajectory data
script_directory = os.path.dirname(os.path.abspath(__file__))
relative_folder = 'trajectories'
target_folder = os.path.join(script_directory, relative_folder)
os.makedirs(target_folder, exist_ok=True)


def saveObject(tau, fname):
    with open(fname, 'wb') as f:
        pickle.dump(tau, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=str, default='Lift')
    parser.add_argument(
        '--robots',
        nargs='+',
        type=str,
        default='Panda',
        help='Which robot(s) to use in the env',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='default',
        help='Specified environment configuration if necessary',
    )
    parser.add_argument(
        '--arm',
        type=str,
        default='right',
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        '--switch-on-grasp',
        action='store_true',
        help='Switch gripper control on gripper action',
    )
    parser.add_argument(
        '--toggle-camera-on-grasp',
        action='store_true',
        help='Switch camera angle on gripper action',
    )
    parser.add_argument(
        '--controller',
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples) or None to get the robot's default controller if it exists",
    )
    parser.add_argument('--device', type=str, default='keyboard')
    parser.add_argument(
        '--pos-sensitivity',
        type=float,
        default=1.0,
        help='How much to scale position user inputs',
    )
    parser.add_argument(
        '--rot-sensitivity',
        type=float,
        default=1.0,
        help='How much to scale rotation user inputs',
    )
    parser.add_argument(
        '--max_fr',
        default=20,
        type=int,
        help='Sleep when simluation runs faster than specified frame rate; 20 fps is real time.',
    )
    parser.add_argument(
        '--reverse_xy',
        type=bool,
        default=False,
        help='(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)',
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    # Create argument configuration
    config = {
        'env_name': args.environment,
        'robots': args.robots,
        'controller_configs': controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if 'TwoArm' in args.environment:
        config['env_configuration'] = args.config
    else:
        args.config = None

    # Create custom placement_initializer
    custom_sampler = UniformRandomSampler(
        name='CustomSampler',
        mujoco_objects=None,
        x_range=[0.0, 0.0],
        y_range=[0.0, 0.0],
        rotation=0.0,
        rotation_axis='z',
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((0, 0, 0.8)),
        z_offset=0.01,
    )

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera='agentview',
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
        placement_initializer=custom_sampler,
        initialization_noise={'magnitude': None, 'type': 'gaussian'},
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})

    # initialize device
    if args.device == 'keyboard':
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        env.viewer.add_keypress_callback(device.on_press)

    try:
        while True:
            # Reset the environment
            obs = env.reset()
            fname = f'{target_folder}/traj_{len(glob.glob(os.path.join(target_folder, "*.pkl")))}.pkl'
            tau = []

            # Setup rendering
            cam_id = 0
            num_cam = len(env.sim.model.camera_names)
            env.render()

            # Initialize variables that should the maintained between resets
            last_grasp = 0

            # Initialize device control
            device.start_control()
            all_prev_gripper_actions = [
                {
                    f'{robot_arm}_gripper': np.repeat(
                        [-1], robot.gripper[robot_arm].dof
                    )
                    for robot_arm in robot.arms
                    if robot.gripper[robot_arm].dof > 0
                }
                for robot in env.robots
            ]

            # Loop until we get a reset from the input or the task completes
            while True:
                start = time.time()

                # Set active robot
                active_robot = env.robots[device.active_robot]

                # Get the newest action
                input_ac_dict = device.input2action()

                # If action is none, then this a reset so we should break
                if input_ac_dict is None:
                    if len(tau) > 1:
                        saveObject(tau, fname)
                    break

                from copy import deepcopy

                action_dict = deepcopy(input_ac_dict)  # {}
                # set arm actions
                for arm in active_robot.arms:
                    if isinstance(
                        active_robot.composite_controller, WholeBody
                    ):  # input type passed to joint_action_policy
                        controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                    else:
                        controller_input_type = active_robot.part_controllers[
                            arm
                        ].input_type

                    if controller_input_type == 'delta':
                        action_dict[arm] = input_ac_dict[f'{arm}_delta']
                    elif controller_input_type == 'absolute':
                        action_dict[arm] = input_ac_dict[f'{arm}_abs']
                    else:
                        raise ValueError

                # keep a no-action reference
                no_action = np.zeros((7,))
                no_action[-1] = all_prev_gripper_actions[0][
                    f'{env.robots[0].arms[0]}_gripper'
                ][0]

                # Maintain gripper state for each robot but only update the active robot with action
                env_action = [
                    robot.create_action_vector(all_prev_gripper_actions[i])
                    for i, robot in enumerate(env.robots)
                ]
                env_action[
                    device.active_robot
                ] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)
                for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                    all_prev_gripper_actions[device.active_robot][
                        gripper_ac
                    ] = action_dict[gripper_ac]

                env.step(env_action)
                env.render()

                """
                # NOTE: This currently only stores state and action if the current action is different from last. To change to storing at every timestep, comment out if condition and uncomment "if True:"
                """
                # store state and action if action taken
                if not np.array_equal(env_action, no_action):
                    # if True:
                    obs_dict = env.observation_spec()
                    observation = [*obs_dict['robot0_proprio-state'], *obs_dict['object-state']]
                    observation = np.array(observation, dtype=np.float32)
                    print(f'env action: {env_action}')
                    tau.append((observation, env_action))

                # limit frame rate if necessary
                if args.max_fr is not None:
                    elapsed = time.time() - start
                    diff = 1 / args.max_fr - elapsed
                    if diff > 0:
                        time.sleep(diff)
    except KeyboardInterrupt:
        if len(tau) > 1:
            saveObject(tau, fname)
