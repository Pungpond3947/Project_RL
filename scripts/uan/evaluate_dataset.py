"""Evaluate trained UAN checkpoints by replaying hardware dataset transitions."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate UAN checkpoints against recorded one-leg hardware data.")
parser.add_argument("--task", type=str, default="oneleg-uan", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel replay windows.")
parser.add_argument("--steps", type=int, default=500, help="Replay steps per batch. Clipped before timeout reset.")
parser.add_argument("--batches", type=int, default=1, help="Number of independent replay batches.")
parser.add_argument("--seed", type=int, default=42, help="Seed for selecting hardware replay starts.")
parser.add_argument(
    "--checkpoints",
    nargs="*",
    default=None,
    help="Checkpoint paths. If omitted, the latest 3 one_leg_uan runs are evaluated.",
)
parser.add_argument("--latest_runs", type=int, default=3, help="How many latest runs to use when checkpoints are omitted.")
parser.add_argument("--log_root", type=str, default="logs/rsl_rl/one_leg_uan", help="Directory containing UAN runs.")
parser.add_argument("--output", type=str, default=None, help="CSV output path.")
parser.add_argument(
    "--dataset_paths",
    nargs="*",
    default=None,
    help="Optional CSV paths to evaluate on. If omitted, the task cfg dataset_paths are used.",
)
parser.add_argument("--plot", action="store_true", default=False, help="Save real-vs-sim joint position plots.")
parser.add_argument("--plot_steps", type=int, default=None, help="Replay steps used for the position plot.")
parser.add_argument("--plot_env_index", type=int, default=0, help="Environment index to plot from the replay batch.")
parser.add_argument("--plot_output", type=str, default=None, help="PNG path for the position plot.")
parser.add_argument("--plot_csv", type=str, default=None, help="CSV path for plotted position traces.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Optional evaluation episode length override. Keep it longer than --steps * env_dt.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import physical_locomotion_ai.tasks  # noqa: E402, F401


def main():
    checkpoints = _resolve_checkpoints(args_cli.checkpoints, Path(args_cli.log_root), args_cli.latest_runs)
    if not checkpoints:
        raise RuntimeError("No checkpoints found to evaluate.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.dataset_paths is not None:
        env_cfg.dataset_paths = tuple(args_cli.dataset_paths)
    env_cfg.seed = args_cli.seed

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    task = env.unwrapped

    eval_steps = min(args_cli.steps, env.max_episode_length - 2)
    if eval_steps < 1:
        raise RuntimeError("Evaluation needs at least one step. Increase episode_length_s.")
    if eval_steps != args_cli.steps:
        print(f"[WARN] Clipped --steps from {args_cli.steps} to {eval_steps} to avoid timeout resets.")
    if args_cli.plot_env_index < 0 or args_cli.plot_env_index >= args_cli.num_envs:
        raise ValueError(f"--plot_env_index must be in [0, {args_cli.num_envs - 1}], got {args_cli.plot_env_index}.")
    plot_steps = min(args_cli.plot_steps if args_cli.plot_steps is not None else eval_steps, eval_steps)

    starts = _make_start_batches(task, args_cli.batches, args_cli.num_envs, args_cli.seed)

    print("[INFO] Evaluating against hardware dataset")
    print(f"[INFO]   num_envs={args_cli.num_envs}, batches={args_cli.batches}, steps={eval_steps}")
    print(f"[INFO]   checkpoints={len(checkpoints)}")
    for checkpoint in checkpoints:
        print(f"[INFO]     {checkpoint}")

    results = []
    results.append(_evaluate_case(env, "baseline_zero_residual", None, starts, eval_steps))
    plot_traces = []
    if args_cli.plot:
        plot_traces.append(
            _collect_position_trace(
                env,
                "baseline_zero_residual",
                None,
                starts[0],
                plot_steps,
                args_cli.plot_env_index,
            )
        )

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    for checkpoint in checkpoints:
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=task.device)
        name = f"{checkpoint.parent.name}:{checkpoint.stem}"
        results.append(_evaluate_case(env, name, policy, starts, eval_steps, checkpoint=checkpoint))
        if args_cli.plot:
            plot_traces.append(
                _collect_position_trace(
                    env,
                    name,
                    policy,
                    starts[0],
                    plot_steps,
                    args_cli.plot_env_index,
                    checkpoint=checkpoint,
                )
            )

    _print_results(results)
    output_path = _write_results(results, args_cli.output)
    print(f"[INFO] Wrote CSV: {output_path}")
    if args_cli.plot:
        plot_path, trace_path = _write_position_plot(plot_traces, output_path, args_cli.plot_output, args_cli.plot_csv)
        print(f"[INFO] Wrote position plot: {plot_path}")
        print(f"[INFO] Wrote position traces: {trace_path}")

    env.close()


def _evaluate_case(env, name, policy, start_batches, steps, checkpoint: Path | None = None):
    task = env.unwrapped
    device = task.device
    joint_count = len(task.actuated_dof_indices)
    source_count = len(task.dataset.source_files)

    q_sq = torch.zeros(joint_count, dtype=torch.float64, device=device)
    q_abs = torch.zeros_like(q_sq)
    dq_sq = torch.zeros_like(q_sq)
    dq_abs = torch.zeros_like(q_sq)
    action_sq = torch.zeros(joint_count, dtype=torch.float64, device=device)
    source_q_sq = torch.zeros(source_count, joint_count, dtype=torch.float64, device=device)
    source_dq_sq = torch.zeros_like(source_q_sq)
    source_counts = torch.zeros(source_count, dtype=torch.float64, device=device)
    reward_sum = torch.zeros((), dtype=torch.float64, device=device)
    count = 0

    with torch.inference_mode():
        for starts in start_batches:
            task.reset_to_dataset_starts(starts)
            task.scene.write_data_to_sim()
            task.sim.forward()
            obs, _ = env.get_observations()

            for _ in range(steps):
                if policy is None:
                    actions = torch.zeros(task.num_envs, joint_count, dtype=torch.float32, device=device)
                else:
                    actions = torch.clamp(policy(obs), -1.0, 1.0)

                obs, rewards, dones, _ = env.step(actions)
                if torch.any(dones):
                    raise RuntimeError("Evaluation hit an environment reset. Reduce --steps or increase episode length.")

                data_indices = task.step_data_indices
                sim_q = task.robot.data.joint_pos[:, task.actuated_dof_indices]
                sim_dq = task.robot.data.joint_vel[:, task.actuated_dof_indices]
                q_error = task.dataset.q_next[data_indices] - sim_q
                dq_error = task.dataset.dq_next[data_indices] - sim_dq

                q_sq += torch.sum(torch.square(q_error.double()), dim=0)
                q_abs += torch.sum(torch.abs(q_error.double()), dim=0)
                dq_sq += torch.sum(torch.square(dq_error.double()), dim=0)
                dq_abs += torch.sum(torch.abs(dq_error.double()), dim=0)
                action_sq += torch.sum(torch.square(actions.double()), dim=0)
                reward_sum += torch.sum(rewards.double())
                count += task.num_envs

                source_ids = task.dataset.source_id[data_indices]
                for source_index in range(source_count):
                    mask = source_ids == source_index
                    source_n = int(torch.sum(mask).item())
                    if source_n == 0:
                        continue
                    source_q_sq[source_index] += torch.sum(torch.square(q_error[mask].double()), dim=0)
                    source_dq_sq[source_index] += torch.sum(torch.square(dq_error[mask].double()), dim=0)
                    source_counts[source_index] += source_n

    q_rmse = torch.sqrt(q_sq / count).cpu().numpy()
    dq_rmse = torch.sqrt(dq_sq / count).cpu().numpy()
    q_mae = (q_abs / count).cpu().numpy()
    dq_mae = (dq_abs / count).cpu().numpy()
    action_rms = torch.sqrt(action_sq / count).cpu().numpy()

    source_metrics = {}
    source_files = [Path(path).stem for path in task.dataset.source_files]
    for source_index, source_name in enumerate(source_files):
        denom = max(float(source_counts[source_index].item()), 1.0)
        source_metrics[f"{source_name}_q_rmse_mean"] = float(torch.mean(torch.sqrt(source_q_sq[source_index] / denom)))
        source_metrics[f"{source_name}_dq_rmse_mean"] = float(torch.mean(torch.sqrt(source_dq_sq[source_index] / denom)))

    return {
        "name": name,
        "checkpoint": "" if checkpoint is None else str(checkpoint),
        "iter": "" if checkpoint is None else _checkpoint_iter(checkpoint),
        "transitions": count,
        "q_rmse_mean": float(np.mean(q_rmse)),
        "dq_rmse_mean": float(np.mean(dq_rmse)),
        "q_mae_mean": float(np.mean(q_mae)),
        "dq_mae_mean": float(np.mean(dq_mae)),
        "reward_mean": float((reward_sum / count).item()),
        "action_rms_mean": float(np.mean(action_rms)),
        "q_rmse_0": float(q_rmse[0]),
        "q_rmse_1": float(q_rmse[1]),
        "q_rmse_2": float(q_rmse[2]),
        "dq_rmse_0": float(dq_rmse[0]),
        "dq_rmse_1": float(dq_rmse[1]),
        "dq_rmse_2": float(dq_rmse[2]),
        **source_metrics,
    }


def _collect_position_trace(
    env,
    name: str,
    policy,
    starts: torch.Tensor,
    steps: int,
    env_index: int,
    checkpoint: Path | None = None,
) -> dict:
    """Replay one batch and record real/sim joint positions for one environment."""
    task = env.unwrapped
    device = task.device
    joint_count = len(task.actuated_dof_indices)
    joint_names = [task.robot.joint_names[index] for index in task.actuated_dof_indices]

    with torch.inference_mode():
        task.reset_to_dataset_starts(starts)
        task.scene.write_data_to_sim()
        task.sim.forward()
        obs, _ = env.get_observations()

        start_index = int(starts[env_index].item())
        source_id = int(task.dataset.source_id[start_index].item())
        source_file = task.dataset.source_files[source_id]
        start_time = float(task.dataset.t[start_index].cpu().item())

        times = [0.0]
        data_rows = [start_index]
        real_q = [task.dataset.q[start_index].detach().cpu().numpy()]
        cmd_q = [task.dataset.cmd[start_index].detach().cpu().numpy()]
        sim_q = [task.robot.data.joint_pos[env_index, task.actuated_dof_indices].detach().cpu().numpy()]
        actions_out = [np.zeros(joint_count, dtype=np.float32)]
        residual_effort = [np.zeros(joint_count, dtype=np.float32)]

        for _ in range(steps):
            if policy is None:
                actions = torch.zeros(task.num_envs, joint_count, dtype=torch.float32, device=device)
            else:
                actions = torch.clamp(policy(obs), -1.0, 1.0)

            obs, _, dones, _ = env.step(actions)
            if torch.any(dones):
                raise RuntimeError("Plot replay hit an environment reset. Reduce --plot_steps or increase episode length.")

            data_index = int(task.step_data_indices[env_index].item())
            next_row = min(data_index + 1, task.dataset.num_rows - 1)
            time_value = float(task.dataset.t[next_row].cpu().item()) - start_time

            times.append(time_value)
            data_rows.append(next_row)
            real_q.append(task.dataset.q_next[data_index].detach().cpu().numpy())
            cmd_q.append(task.dataset.cmd[next_row].detach().cpu().numpy())
            sim_q.append(task.robot.data.joint_pos[env_index, task.actuated_dof_indices].detach().cpu().numpy())
            actions_out.append(actions[env_index].detach().cpu().numpy())
            residual_effort.append((actions[env_index] * task.cfg.residual_effort_scale).detach().cpu().numpy())

    return {
        "name": name,
        "checkpoint": "" if checkpoint is None else str(checkpoint),
        "source_file": source_file,
        "joint_names": joint_names,
        "time": np.asarray(times, dtype=np.float64),
        "data_row": np.asarray(data_rows, dtype=np.int64),
        "real_q": np.asarray(real_q, dtype=np.float32),
        "cmd_q": np.asarray(cmd_q, dtype=np.float32),
        "sim_q": np.asarray(sim_q, dtype=np.float32),
        "action": np.asarray(actions_out, dtype=np.float32),
        "residual_effort": np.asarray(residual_effort, dtype=np.float32),
    }


def _make_start_batches(task, batches: int, num_envs: int, seed: int):
    rng = np.random.default_rng(seed)
    batches_out = []
    valid_count = task.dataset.num_valid_starts
    for _ in range(batches):
        choices = rng.integers(0, valid_count, size=num_envs, endpoint=False)
        choice_tensor = torch.as_tensor(choices, dtype=torch.long, device=task.device)
        batches_out.append(task.dataset.valid_start_indices[choice_tensor])
    return batches_out


def _resolve_checkpoints(raw_checkpoints: list[str] | None, log_root: Path, latest_runs: int) -> list[Path]:
    if raw_checkpoints:
        return [_resolve_path(path) for path in raw_checkpoints]

    root = _resolve_path(str(log_root))
    run_dirs = [path for path in root.iterdir() if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime)

    checkpoints = []
    for run_dir in run_dirs:
        models = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_iter)
        if models:
            checkpoints.append(models[-1])
    return checkpoints[-latest_runs:]


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _checkpoint_iter(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _print_results(results: list[dict]):
    baseline = results[0]
    header = (
        f"{'case':<42} {'iter':>6} {'q_rmse':>10} {'dq_rmse':>10} "
        f"{'reward':>10} {'act_rms':>9} {'q_impr':>9} {'dq_impr':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for result in results:
        q_impr = _improvement(baseline["q_rmse_mean"], result["q_rmse_mean"])
        dq_impr = _improvement(baseline["dq_rmse_mean"], result["dq_rmse_mean"])
        print(
            f"{result['name']:<42} {str(result['iter']):>6} "
            f"{result['q_rmse_mean']:>10.5f} {result['dq_rmse_mean']:>10.5f} "
            f"{result['reward_mean']:>10.5f} {result['action_rms_mean']:>9.5f} "
            f"{q_impr:>8.1f}% {dq_impr:>8.1f}%"
        )


def _improvement(reference: float, value: float) -> float:
    if reference == 0.0:
        return 0.0
    return 100.0 * (reference - value) / reference


def _write_position_plot(
    traces: list[dict],
    results_output_path: Path,
    plot_output: str | None,
    trace_output: str | None,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = _resolve_path(plot_output) if plot_output is not None else _with_suffix(results_output_path, "_positions.png")
    trace_path = _resolve_path(trace_output) if trace_output is not None else _with_suffix(results_output_path, "_positions.csv")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    joint_names = traces[0]["joint_names"]
    for trace in traces:
        q_error = trace["real_q"] - trace["sim_q"]
        trace["q_rmse"] = np.sqrt(np.mean(np.square(q_error), axis=0))
        trace["q_rmse_mean"] = float(np.mean(trace["q_rmse"]))

    fig, axes = plt.subplots(len(joint_names), 1, figsize=(13.5, 8), sharex=True)
    if len(joint_names) == 1:
        axes = [axes]

    real_time = traces[0]["time"]
    real_q = traces[0]["real_q"]
    cmd_q = traces[0]["cmd_q"]
    for joint_index, joint_name in enumerate(joint_names):
        ax = axes[joint_index]
        ax.plot(
            real_time,
            cmd_q[:, joint_index],
            color="red",
            linestyle="-",
            linewidth=0.8,
            label="command",
        )
        ax.plot(real_time, real_q[:, joint_index], color="black", linewidth=2.0, label="real")
        for trace in traces:
            line_style = "--" if trace["name"] == "baseline_zero_residual" else "-"
            label = f"{trace['name']} (mean RMSE={trace['q_rmse_mean']:.5f} rad)"
            ax.plot(trace["time"], trace["sim_q"][:, joint_index], linestyle=line_style, linewidth=0.5, label=label)
        ax.set_ylabel(f"{joint_name}\n(rad)")
        ax.grid(True, alpha=0.3)
        rmse_text = "RMSE (rad)\n" + "\n".join(
            f"{_short_case_name(trace['name'])}: {trace['q_rmse'][joint_index]:.5f}" for trace in traces
        )
        ax.text(
            1.01,
            0.5,
            rmse_text,
            transform=ax.transAxes,
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )
        if joint_index == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Real vs Sim Joint Positions\nsource: {Path(traces[0]['source_file']).name}")
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 0.93])
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    with trace_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case",
                "checkpoint",
                "source_file",
                "step",
                "data_row",
                "time",
                "joint",
                "cmd_q",
                "real_q",
                "sim_q",
                "q_error",
                "q_rmse_joint",
                "q_rmse_mean",
                "action",
                "residual_effort",
            ],
        )
        writer.writeheader()
        for trace in traces:
            for step_index, time_value in enumerate(trace["time"]):
                for joint_index, joint_name in enumerate(trace["joint_names"]):
                    real_value = float(trace["real_q"][step_index, joint_index])
                    cmd_value = float(trace["cmd_q"][step_index, joint_index])
                    sim_value = float(trace["sim_q"][step_index, joint_index])
                    writer.writerow(
                        {
                            "case": trace["name"],
                            "checkpoint": trace["checkpoint"],
                            "source_file": trace["source_file"],
                            "step": step_index,
                            "data_row": int(trace["data_row"][step_index]),
                            "time": float(time_value),
                            "joint": joint_name,
                            "cmd_q": cmd_value,
                            "real_q": real_value,
                            "sim_q": sim_value,
                            "q_error": real_value - sim_value,
                            "q_rmse_joint": float(trace["q_rmse"][joint_index]),
                            "q_rmse_mean": float(trace["q_rmse_mean"]),
                            "action": float(trace["action"][step_index, joint_index]),
                            "residual_effort": float(trace["residual_effort"][step_index, joint_index]),
                        }
                    )

    return plot_path, trace_path


def _short_case_name(name: str) -> str:
    if name == "baseline_zero_residual":
        return "baseline"
    if ":" in name:
        run_name, model_name = name.split(":", maxsplit=1)
        return f"{run_name[-8:]}:{model_name}"
    return name


def _with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}")


def _write_results(results: list[dict], output: str | None) -> Path:
    if output is None:
        out_dir = Path("logs/rsl_rl/one_leg_uan/eval_dataset")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    else:
        output_path = _resolve_path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for result in results for key in result.keys()})
    preferred = [
        "name",
        "iter",
        "checkpoint",
        "transitions",
        "q_rmse_mean",
        "dq_rmse_mean",
        "q_mae_mean",
        "dq_mae_mean",
        "reward_mean",
        "action_rms_mean",
        "q_rmse_0",
        "q_rmse_1",
        "q_rmse_2",
        "dq_rmse_0",
        "dq_rmse_1",
        "dq_rmse_2",
    ]
    ordered = preferred + [key for key in fieldnames if key not in preferred]

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(results)
    return output_path


if __name__ == "__main__":
    main()
    simulation_app.close()
