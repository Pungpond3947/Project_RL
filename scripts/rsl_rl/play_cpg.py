"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import physical_locomotion_ai.tasks  # noqa: F401



### For plotting and logging
import pandas as pd
import csv
from collections import deque
import matplotlib.pyplot as plt
import numpy as np



def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    log_dir = "/home/isaac_drl/physicalAI_2026/logs/rsl_rl/one_leg_cpg"
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model



    timestep = 0
    # reset environment
    obs, _ = env.get_observations()
    
    # Foot contact sensor info
    robot = env.unwrapped.robot
    ee_id = robot.body_names.index("end_effector")


    # ======= plotting setup ========
    WINDOW = 100  
    ENV_I = 0  # which env to plot (0 is fine if num_envs=1)

    # Plotting keys (from obs_extras) !!!!
    max_timestep = 300

    obs_buff = np.zeros((max_timestep,13)) # 500,13
    pos_buff = np.zeros((max_timestep,3))
    vel_buff = np.zeros((max_timestep,3))
    act_buff = np.zeros((max_timestep,3))
    act_clip_buff = np.zeros((max_timestep,3))
    vel_buff = np.zeros((max_timestep,3))
    torque_buff = np.zeros((max_timestep,3))
    pos_raw_buff = np.zeros((max_timestep,3))
    acc_buf = np.zeros((max_timestep,3))
    fc_buff = np.zeros((max_timestep,1))
    phase_buff = np.zeros((max_timestep,1))
    tip_vel_buff = np.zeros((max_timestep,1))
    body_vel_buff = np.zeros((max_timestep,1))
    body_accel_buff = np.zeros((max_timestep,1))

    # Action space
    action = torch.zeros((env.num_envs, 3), device=env.device)

    # simulate environment
    # while simulation_app.is_running():
    for i in range(max_timestep+1):
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            action[:, 0] = math.sin(0.10 * i)
            action[:, 1] = 0.2 *math.cos(0.10 * i) - 0.1
            action[:, 2] = 0.025*math.cos(0.10 * i) - 0.15

            actions = action
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            
            if timestep < max_timestep:
                # print(obs.shape, actions.shape)
                obs_buff[timestep] = obs[ENV_I,:].detach().cpu().numpy()
                pos_buff[timestep] = infos['obs_extras']['pos_raw'][ENV_I].detach().cpu().numpy()
                vel_buff[timestep] = infos["obs_extras"]['vel_raw'][ENV_I].detach().cpu().numpy()
                act_clip_buff[timestep] = infos["obs_extras"]['actions_clip'][ENV_I].detach().cpu().numpy()
                phase_buff[timestep] = infos["obs_extras"]['phase_frc'][ENV_I].detach().cpu().numpy()

                torque_buff[timestep] = infos["obs_extras"]['torque_raw'][ENV_I].detach().cpu().numpy() 
                act_buff[timestep] = actions[ENV_I,:].detach().cpu().numpy()
                pos_raw_buff[timestep] = infos["obs_extras"]['pos_raw'][ENV_I].detach().cpu().numpy()
                # acc_buf[timestep] = infos["obs_extras"]['accel_raw'][ENV_I].detach().cpu().numpy()
                fc_buff[timestep] = infos["obs_extras"]['total_contact_force'][ENV_I].detach().cpu().numpy()
                tip_vel_buff[timestep] = infos["obs_extras"]['tip_vel'][ENV_I].detach().cpu().numpy()
                # body_vel_buff[timestep] = infos["obs_extras"]['body_vel'][ENV_I].detach().cpu().numpy()
                # body_accel_buff[timestep] = infos["obs_extras"]['body_accel'][ENV_I].detach().cpu().numpy()
                print(f"Step {timestep}")
            print(actions[ENV_I,:])
            if timestep == max_timestep-1:
                # np.save("obs.npy",obs_buff)
                np.save("pos_raw.npy",pos_raw_buff)
            #     np.save("act.npy",act_buff)
            #     print("Saved obs and actions to obs.npy and act.npy")


            timestep += 1
    env.close()
    
    plt.figure(figsize=(12, 8))
    plt.subplot(6, 2, 1)
    plt.plot(pos_buff[:timestep, 0], label='Pos 0')
    # plt.plot(act_buff[:timestep, 0], label='Act 0')
    plt.plot(act_clip_buff[:timestep, 0], label='Act Clip 0')
    plt.title('joint 1')
    plt.legend()

    plt.subplot(6, 2, 2)
    plt.plot(pos_buff[:timestep, 1], label='Pos 1')
    # plt.plot(act_buff[:timestep, 1], label='Act 1')
    plt.plot(act_clip_buff[:timestep, 1], label='Act Clip 1')
    plt.title('joint 2')
    plt.legend()

    plt.subplot(6, 2, 3)
    plt.plot(pos_buff[:timestep, 2], label='Pos 2')
    # plt.plot(act_buff[:timestep, 2], label='Act 2')
    plt.plot(act_clip_buff[:timestep, 2], label='Act Clip 2')
    plt.title('joint 3')
    plt.legend()
    
    plt.subplot(6, 2, 4)
    plt.plot(vel_buff[:timestep, 0], label='vel 0')
    plt.plot(vel_buff[:timestep, 1], label='vel 1')
    plt.plot(vel_buff[:timestep, 2], label='vel 2')
    plt.title('Velocities')
    plt.legend()

    plt.subplot(6, 2, 5)
    plt.plot(fc_buff[:timestep, 0], label='Contact Force')
    plt.title('Contact Force')
    plt.legend()



    plt.subplot(6, 2, 6)
    plt.plot(torque_buff[:timestep, 0], label='Torque 0')
    plt.plot(torque_buff[:timestep, 1], label='Torque 1')
    plt.plot(torque_buff[:timestep, 2], label='Torque 2')
    plt.title('Torques')
    plt.legend()

    plt.subplot(6, 2, 7)
    plt.plot(phase_buff[:timestep, 0], label='Phase')
    plt.title('Phase Force penalty')
    plt.legend()



    plt.subplot(6, 2, 9)
    plt.plot(tip_vel_buff[:timestep, 0], label='End Effector Velocity')
    plt.title('End Effector Velocity')
    plt.legend()

    plt.subplot(6, 2, 10)
    plt.plot(body_vel_buff[:timestep, 0], label='Body Velocity')
    plt.title('Body Velocity')
    plt.legend()

    # plt.subplot(6, 2, 11)
    # plt.plot(acc_buf[:timestep, 0], label='Acc 0')
    # plt.plot(acc_buf[:timestep, 1], label='Acc 1')
    # plt.plot(acc_buf[:timestep, 2], label='Acc 2')
    # plt.title('Accelerations')
    # plt.legend()

    # plt.subplot(6, 2, 12)
    # plt.plot(body_accel_buff[:timestep, 0], label='Body Acceleration')
    # plt.title('Body Acceleration')
    # plt.legend()    

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
