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
    watch_keys = [
        # "phase_sign",
        "total_contact_force",
        "phase force",
        # "tip_vel",
        # "body_vel",
        "phase",
        "episode_length",
        "weight_phase_scale",
        "weight_stand_target_scale",
        "stand_target",
    ]

    nrows, ncols = 3, 3  
    buf = {k: deque(maxlen=WINDOW) for k in watch_keys}
    tbuf = deque(maxlen=WINDOW)
    plt.ion()
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True)
    axs = np.array(axs).reshape(-1)  # flatten
    lines = {}
    for i, k in enumerate(watch_keys):
        ax = axs[i]
        (ln,) = ax.plot([], [])
        ax.set_title(k)
        ax.grid(True, alpha=0.3)
        lines[k] = ln

    for j in range(len(watch_keys), nrows * ncols):
        axs[j].axis("off")

    fig.suptitle(f"Rolling {WINDOW}-step signals (env {ENV_I})")
    fig.tight_layout()

    obs_buff = np.zeros((500,13))
    act_buff = np.zeros((500,3))
    
    # === Write CSV header ===
    with open(steps_path, "w", newline="") as steps_f:
        steps_w = csv.writer(steps_f)

        # header
        steps_w.writerow([
            "episode_id",
            "timestep",
            "phase_reward", "timeout_reward", "goal_reward", "distance_reward", "total_reward",
            "pos0", "pos1", "pos2",
            "vel0", "vel1", "vel2",
            "a0", "a1", "a2",
            "robot_pos_x",
            "phase",
            "foot_pos_x", "foot_pos_y", "foot_pos_z"    
        ])
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, rewards, dones, infos = env.step(actions)
                
                if timestep < 500:
                    # print(obs.shape, actions.shape)
                    obs_buff[timestep] = obs[0,:].detach().cpu().numpy()
                    act_buff[timestep] = actions[0,:].detach().cpu().numpy()
                    # print(obs_buff.shape , act_buff.shape)
                    print(f"Step {timestep}")
                if timestep == 499:
                    np.save("obs.npy",obs_buff)
                    np.save("act.npy",act_buff)
                    print("Saved obs and actions to obs.npy and act.npy")

                # print(f"{infos['time_outs']=}, {dones=}")

                # dict_keys(['obs_extras', 'log', 'observations', 'time_outs'])
                # print(infos['obs_extras'].keys())

                # ---- extract tensors (assume num_envs=1 for evaluation) ----
                lg = infos.get("log", {})
                ex = infos.get("obs_extras", {})

                tbuf.append(timestep)

                for k in watch_keys:
                    if k in ex:
                        buf[k].append(to_scalar(ex[k], env_i=ENV_I))
                    else:
                        buf[k].append(np.nan)


                # ---------------------------------- Visualize




                # REDRAW_EVERY = 5  # increase if slow
                # if timestep % REDRAW_EVERY == 0 and len(tbuf) > 2:
                #     x = np.arange(len(tbuf))

                #     for k in watch_keys:
                #         y = np.asarray(buf[k], dtype=np.float32)
                #         ax = lines[k].axes
                #         lines[k].set_data(x, y)

                #         # per-axis autoscale (robust)
                #         yf = y[np.isfinite(y)]
                #         if yf.size > 0:
                #             y_min, y_max = float(yf.min()), float(yf.max())
                #             pad = 0.05 * (y_max - y_min + 1e-6)
                #             ax.set_ylim(y_min - pad, y_max + pad)
                #         ax.set_xlim(0, max(len(tbuf) - 1, 1))

                #     fig.canvas.draw()
                #     fig.canvas.flush_events()
                #     plt.pause(0.001)

                # # log rewards (scalars)
                # phase_r   = _to_float(lg.get("phase_reward", 0.0))
                # timeout_r = _to_float(lg.get("timeout_reward", 0.0))
                # goal_r    = _to_float(lg.get("goal_reward", 0.0))
                # dist_r    = _to_float(lg.get("distance_reward", 0.0))
                # total_r   = _to_float(lg.get("total_reward", 0.0))

                # # obs extras (vectors / scalars)
                # pos_scale = ex["pos_scale"][0]        # shape (3,)
                # vel_scale = ex["vel_scale"][0]        # shape (3,)
                # prev_a    = ex["prev_actions"][0]     # shape (3,)
                # robot_x   = ex["robot_pos_x"][0]      # shape (1,) or scalar
                # phase     = ex["phase"][0]            # shape (1,) or scalar

                # # convert to floats
                # p = pos_scale.detach().cpu().tolist()
                # v = vel_scale.detach().cpu().tolist()
                # a = prev_a.detach().cpu().tolist()

                # # robot_x / phase
                # rx = float(robot_x.detach().cpu().view(-1)[0].item())
                # ph = float(phase.detach().cpu().view(-1)[0].item())

                # foot_pos = robot.data.body_pos_w[:, ee_id, :][0].detach().cpu().tolist()

                # fx, fy, fz = foot_pos
                # # print(f"Foot position2: x={fx}, y={fy}, z={fz}")

                # # ========== write Logging =========
                # # write row
                # steps_w.writerow([
                #     episode_count,
                #     timestep,
                #     phase_r, timeout_r, goal_r, dist_r, total_r,
                #     p[0], p[1], p[2],
                #     v[0], v[1], v[2],
                #     a[0], a[1], a[2],
                #     rx,
                #     ph,
                #     foot_pos[0], foot_pos[1], foot_pos[2]     
                # ])

                # ---------------------------------- Visualize



                timestep += 1

                # if dones.any():
                #     episode_count += 1
                #     print(f"[INFO] Episode {episode_count} finished")
                #     if episode_count >= 2:
                #         break

    # close the simulator
    env.close()

    print(f"[INFO] CSV safely saved to: {steps_path}")



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
