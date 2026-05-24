"""Show the one-leg robot home pose in Isaac Sim."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root_height", type=float, default=0.7, help="Robot root z position for viewing.")
parser.add_argument("--linear", type=float, default=0.1, help="linear_left_right home position.")
parser.add_argument("--hip", type=float, default=0.0, help="hip_joint home position in radians.")
parser.add_argument("--knee", type=float, default=0.0, help="knee_joint home position in radians.")
parser.add_argument("--ankle", type=float, default=0.0, help="ankle_joint home position in radians.")
parser.add_argument(
    "--free_physics",
    action="store_true",
    default=False,
    help="Let physics move the robot. By default the exact home pose is held for visual inspection.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from physical_locomotion_ai.assets.models.one_leg import ONE_LEG_CFG  # noqa: E402


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.set_camera_view(eye=[2.0, 2.0, 1.4], target=[0.0, 0.0, 0.45])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = ONE_LEG_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, args_cli.root_height)
    robot_cfg.init_state.joint_pos = {
        "linear_left_right": args_cli.linear,
        "hip_joint": args_cli.hip,
        "knee_joint": args_cli.knee,
        "ankle_joint": args_cli.ankle,
    }
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    robot.update(sim.get_physics_dt())

    print("\n[HOME] one-leg sim home pose")
    for index, name in enumerate(robot.joint_names):
        value = float(joint_pos[0, index].cpu())
        print(f"  {index:>2}  {name:<24} {value:> .6f} rad   {math.degrees(value):> .3f} deg")
    print("\n[INFO] GUI is holding this pose. Close Isaac Sim to exit.\n")

    while simulation_app.is_running():
        if not args_cli.free_physics:
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
        else:
            robot.set_joint_position_target(joint_pos)
            robot.write_data_to_sim()

        sim.step()
        robot.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
