"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

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
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    # resume_path = get_checkpoint_path(agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        # ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        # ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )


    
    def to_scalar(x, env_i=0):
        """Convert a tensor/array/scalar from infos to a python float (for one env)."""
        if torch.is_tensor(x):
            x = x.detach()
            # pick env index if it has env dimension
            if x.ndim >= 1 and x.shape[0] > env_i:
                x = x[env_i]
            # reduce any remaining dims
            return float(x.reshape(-1)[0].cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.ravel(x)[0])
        return float(x)

    # where to save
    export_dir = os.path.join(log_dir, "eval_csv")
    os.makedirs(export_dir, exist_ok=True)

    steps_path = os.path.join(export_dir, "steps.csv")

    timestep = 0
    episode_count = 0
    # reset environment
    obs, _ = env.get_observations()
    
    # Foot contact sensor info
    robot = env.unwrapped.robot
    ee_id = robot.body_names.index("end_effector")


    # ======= plotting setup ========
    WINDOW = 100  
    ENV_I = 0  # which env to plot (0 is fine if num_envs=1)

    # Plotting keys (from obs_extras) !!!!

    obs_buff = np.zeros((500,12)) # 500,13
    pos_buff = np.zeros((500,3))
    vel_buff = np.zeros((500,3))
    act_buff = np.zeros((500,3))
    act_clip_buff = np.zeros((500,3))
    vel_buff = np.zeros((500,3))
    torque_buff = np.zeros((500,3))
    pos_raw_buff = np.zeros((500,3))
    fc_buff = np.zeros((500,1))

    # simulate environment
    # while simulation_app.is_running():
    for i in range(501):
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            
            if timestep < 500:
                # print(obs.shape, actions.shape)
                obs_buff[timestep] = obs[0,:].detach().cpu().numpy()
                pos_buff[timestep] = infos['obs_extras']['pos_raw'][0].detach().cpu().numpy()
                vel_buff[timestep] = infos["obs_extras"]['vel_raw'][0].detach().cpu().numpy()
                act_clip_buff[timestep] = infos["obs_extras"]['actions_clip'][0].detach().cpu().numpy()
                fc_buff[timestep] = infos["obs_extras"]['total_contact_force'][ENV_I].detach().cpu().numpy()
                torque_buff[timestep] = infos["obs_extras"]['torque_raw'][0].detach().cpu().numpy() 
                act_buff[timestep] = actions[0,:].detach().cpu().numpy()
                pos_raw_buff[timestep] = infos["obs_extras"]['pos_raw'][0].detach().cpu().numpy()

                print(f"Step {timestep}")
            if timestep == 499:
                # np.save("obs.npy",obs_buff)
                np.save("pos_raw.npy",pos_raw_buff)
            #     np.save("act.npy",act_buff)
            #     print("Saved obs and actions to obs.npy and act.npy")


            timestep += 1
    env.close()
    
    plt.figure(figsize=(12, 8))
    plt.subplot(5, 2, 1)
    plt.plot(pos_buff[:timestep, 0], label='Pos 0')
    plt.plot(act_buff[:timestep, 0], label='Act 0')
    plt.title('joint 1')
    plt.legend()

    plt.subplot(5, 2, 2)
    plt.plot(pos_buff[:timestep, 1], label='Pos 1')
    plt.plot(act_buff[:timestep, 1], label='Act 1')
    plt.title('joint 2')
    plt.legend()

    plt.subplot(5, 2, 3)
    plt.plot(pos_buff[:timestep, 2], label='Pos 2')
    plt.plot(act_buff[:timestep, 2], label='Act 2')
    plt.title('joint 3')
    plt.legend()

    plt.subplot(5, 2, 4)
    plt.plot(pos_buff[:timestep, 0], label='Pos 0')
    plt.plot(act_clip_buff[:timestep, 0], label='Act 0')
    plt.title('joint clipped 1')
    plt.legend()

    plt.subplot(5, 2, 5)
    plt.plot(pos_buff[:timestep, 1], label='Pos 1')
    plt.plot(act_clip_buff[:timestep, 1], label='Act 1')
    plt.title('joint clipped 2')
    plt.legend()

    plt.subplot(5, 2, 6)
    plt.plot(pos_buff[:timestep, 2], label='Pos 2')
    plt.plot(act_clip_buff[:timestep, 2], label='Act 2')
    plt.title('joint clipped 3')
    plt.legend()

    
    plt.subplot(5, 2, 7)
    plt.plot(vel_buff[:timestep, 0], label='vel 0')
    plt.plot(vel_buff[:timestep, 1], label='vel 1')
    plt.plot(vel_buff[:timestep, 2], label='vel 2')
    plt.title('Velocities')
    plt.legend()

    plt.subplot(5, 2, 8)
    plt.plot(torque_buff[:timestep, 0], label='Torque 0')
    plt.plot(torque_buff[:timestep, 1], label='Torque 1')
    plt.plot(torque_buff[:timestep, 2], label='Torque 2')
    plt.title('Torques')
    plt.legend()

    plt.subplot(5, 2, 9)
    plt.plot(act_clip_buff[:timestep, 0], label='Clip Act 0')
    plt.plot(act_clip_buff[:timestep, 1], label='Clip Act 1')
    plt.plot(act_clip_buff[:timestep, 2], label='Clip Act 2')
    plt.title('Clipped Actions')
    plt.legend()

    plt.subplot(5, 2, 10)
    plt.plot(fc_buff[:timestep, 0], label='Contact Force')
    plt.title('Contact Force')
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
