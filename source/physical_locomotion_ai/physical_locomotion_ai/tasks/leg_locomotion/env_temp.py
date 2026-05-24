"""
Fixed One-Leg Point-to-Point Locomotion Environment
====================================================
Key fixes over original:
1. Phase clock wraps properly per gait cycle (in seconds, not raw steps)
2. Simple sin-based stance/swing signals (no fragile sigmoid composition)
3. Fixed reward weights (no dynamic weight scaling)
4. Forward progress reward (not just distance)
5. Proper cycle time (~0.4-0.6s per hop)
6. Clean contact detection without excessive smoothing
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
import isaaclab.utils.math as math_utils

import math
import torch
import numpy as np

from physical_locomotion_ai.assets.models.one_leg import ONE_LEG_CFG
from physical_locomotion_ai.tasks.leg_locomotion.one_leg_env_cfg import *


class OneLegP2PTask(DirectRLEnv):
    """One-legged hopper on a linear rail, trained to reach a goal position."""

    cfg: OneLegEnvCfg

    def __init__(self, cfg: OneLegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Actuated joint indices ────────────────────────────────────────
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            idx = self.robot.joint_names.index(joint_name)
            self.actuated_dof_indices.append(idx)
            print(f"Joint: {joint_name} | Index: {idx}")
        self.actuated_dof_indices.sort()
        self.num_actuated = len(self.actuated_dof_indices)

        # ── Joint info ────────────────────────────────────────────────────
        self._joint_dof_idx, self._joint_names = self.robot.find_joints(".*")
        print(f"[INFO] Joints: {self._joint_names}")
        print(f"[INFO] Bodies: {self.robot.body_names}")
        print(f"[INFO] DOF limits lower: {self.robot.data.soft_joint_pos_limits[0, :, 0]}")
        print(f"[INFO] DOF limits upper: {self.robot.data.soft_joint_pos_limits[0, :, 1]}")
        print(f"[INFO] Default DOF: {self.robot.data.default_joint_pos[0, :]}")

        # ── Precompute body/joint lookup indices (scalar ints) ─────────
        # find_bodies/find_joints return ([idx_list], [name_list])
        # Extract scalar int to avoid extra tensor dimensions when indexing
        self._linear_joint_idx = self.robot.find_joints("linear_left_right")[0]    # list for joint_pos indexing → keeps dim
        self._linear_body_idx = self.robot.find_bodies("linear_left_right_Link")[0][0]  # scalar int
        self._ee_body_idx = self.robot.find_bodies("end_effector")[0][0]                # scalar int

        # ── Normalization limits for obs ──────────────────────────────────
        self.dof_pos_limits = torch.full(
            (self.num_envs, self.num_actuated), math.pi,
            dtype=torch.float32, device=self.sim.device,
        )
        self.dof_vel_limits = torch.full(
            (self.num_envs, self.num_actuated), math.pi,
            dtype=torch.float32, device=self.sim.device,
        )

        # ── Action buffers ────────────────────────────────────────────────
        self.actions = torch.zeros(self.num_envs, self.num_actuated, dtype=torch.float32, device=self.sim.device)
        self.actions_raw = torch.zeros_like(self.actions)  # unfiltered policy output
        self.prev_actions = torch.zeros_like(self.actions)
        self.prev_prev_actions = torch.zeros_like(self.actions)  # for jerk penalty

        # Action smoothing: exponential moving average filter
        # Higher = smoother but more sluggish. 0.2-0.4 works well.
        self.action_ema_alpha = 0.3  # new = alpha * raw + (1 - alpha) * prev_smoothed

        # Max action change per step (radians). Prevents violent jerks.
        self.max_action_delta = 0.15  # ~8.6 deg/step

        # ── Contact state ─────────────────────────────────────────────────
        self.foot_contact_force = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.is_foot_in_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)

        # ── Phase clock state ─────────────────────────────────────────────
        # cycle_time is in SECONDS (not steps!)
        # Initialize to 0.5s default — will be overwritten by _reset_idx, but prevents NaN
        # if _compute_phase_signals runs before first reset
        self.cycle_time = torch.full((self.num_envs,), 0.75, dtype=torch.float32, device=self.sim.device)
        self.phase_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)  # accumulated time
        self.C_frc = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.C_vel = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)

        # Stance ratio: fraction of cycle spent in stance (foot on ground)
        # For a hopper, ~0.3-0.5 is typical (short contact, longer flight)
        self.stance_ratio = 0.4

        # ── Robot state buffers ───────────────────────────────────────────
        self.default_joint = self.robot.data.default_joint_pos
        self.robot_pos_x = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.sim.device)
        self.prev_robot_pos_x = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.sim.device)
        self.robotbody_pos_x = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.sim.device)

        # ── Goal / command ────────────────────────────────────────────────
        self.target_x = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.sim.device)
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.sim.device)
        self.command_obs = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.sim.device)
        self.prev_command_obs = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.sim.device)

        # ── Curriculum: start with short distances, increase over time ────
        self.curriculum_level = 0
        self.curriculum_goals = [1.0, 2.0, 3.0, 5.0]  # max goal distance per level
        self.success_rate_buf = torch.zeros(100, dtype=torch.float32, device=self.sim.device)
        self.success_idx = 0

        # ── Goal visualizer ───────────────────────────────────────────────
        marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
        marker_cfg.markers["arrow"].scale = (0.2, 0.2, 0.2)
        self.goal_visualizer = VisualizationMarkers(marker_cfg)
        self.goal_visualizer.set_visibility(True)

        # ── Logging ───────────────────────────────────────────────────────
        self._episode_reward = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "progress_reward",
                "phase_reward",
                "distance_reward",
                "stand_reward",
                "regularization",
                "energy",
                "total_reward",
            ]
        }

    # ══════════════════════════════════════════════════════════════════════
    # Scene setup
    # ══════════════════════════════════════════════════════════════════════
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_force)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ══════════════════════════════════════════════════════════════════════
    # Physics step
    # ══════════════════════════════════════════════════════════════════════
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_prev_actions = self.prev_actions.clone()
        self.prev_actions = self.actions.clone()

        # Store raw policy output (for logging/debugging)
        self.actions_raw = actions.clone()

        # ── Action smoothing pipeline ─────────────────────────────────────
        # Step 1: EMA filter — blend new action with previous smoothed action
        smoothed = self.action_ema_alpha * actions + (1.0 - self.action_ema_alpha) * self.prev_actions

        # Step 2: Rate limiting — clamp maximum change per step
        # delta = smoothed - self.prev_actions
        # delta = actions.clone() - self.prev_actions  # apply rate limit to raw action for more responsive control
        # delta = torch.clamp(delta, -self.max_action_delta, self.max_action_delta)
        self.actions = self.actions_raw # self.prev_actions + delta

        # Advance phase clock by the simulation step time
        self.phase_time += self.step_dt

    def _apply_action(self):
        self.robot.set_joint_position_target(self.actions, joint_ids=self.actuated_dof_indices)

    # ══════════════════════════════════════════════════════════════════════
    # Phase clock — clean periodic stance/swing signals
    # ══════════════════════════════════════════════════════════════════════
    def _compute_phase_signals(self):
        """
        Compute stance/swing phase signals using smooth sigmoid transitions.

        Phase ∈ [0, 1) wraps every cycle_time seconds.

        C_vel: smooth swing signal built from overlapping sigmoids:
            - start_period : rises at phase=0   (swing begins)
            - end_period   : rises at phase=r   (swing ends)
            - wrap_around  : handles the wrap-around window near phase=1
        C_vel ∈ [-1, 0]  (-1 = full swing, 0 = full stance)

        C_frc: derived directly from C_vel so the two are always consistent:
        C_frc = -(1 + C_vel)  →  -1 during stance, 0 during swing
        """
        kappa = 50.0
        r = self.stance_ratio  # fraction of cycle in stance (0.4 default)

        # Wrap phase to [0, 1)
        phase = (self.phase_time % self.cycle_time) / self.cycle_time  # (num_envs,)

        start_period  = sigmoid(phase,           kappa)
        end_period    = sigmoid(phase - r,       kappa)
        wrap_around   = sigmoid(phase - 1.0,     kappa) \
                    - sigmoid(phase - 1.0 - r, kappa)

        self.C_vel = -torch.clamp(start_period - end_period + wrap_around, min=0.0, max=1.0)
        self.C_frc = (1.0 + self.C_vel) * -1.0

        # Store phase value for obs
        self.phase_value = phase

    # ══════════════════════════════════════════════════════════════════════
    # Contact detection
    # ══════════════════════════════════════════════════════════════════════
    def _update_contact(self):
        """Update foot contact force and binary contact status."""
        net_forces = self.scene["contact_sensor"].data.net_forces_w  # (num_envs, num_bodies, 3)
        force_magnitude = torch.norm(net_forces, dim=-1).squeeze(-1)  # (num_envs,)

        # Light smoothing (0.8 new + 0.2 old) to reduce noise without adding too much lag
        self.foot_contact_force = 0.8 * force_magnitude + 0.2 * self.foot_contact_force
        self.is_foot_in_contact = self.foot_contact_force > 1.0

    # ══════════════════════════════════════════════════════════════════════
    # Observations
    # ══════════════════════════════════════════════════════════════════════
    def _get_observations(self) -> dict:
        # Update contact and phase
        self._update_contact()
        self._compute_phase_signals()

        # Joint state
        dof_pos = self.robot.data.joint_pos[:, self.actuated_dof_indices]
        dof_vel = self.robot.data.joint_vel[:, self.actuated_dof_indices]

        # Normalize
        pos_scaled = math_utils.scale_transform(dof_pos, -self.dof_pos_limits, self.dof_pos_limits)
        vel_scaled = math_utils.scale_transform(dof_vel, -self.dof_vel_limits, self.dof_vel_limits)

        # Robot base position (from linear guide joint)
        self.prev_robot_pos_x = self.robot_pos_x.clone()
        self.robot_pos_x = self.robot.data.joint_pos[:, self._linear_joint_idx]        # (num_envs, 1)
        self.robotbody_pos_x = self.robot.data.body_link_pose_w[:, self._linear_body_idx, 0:1]  # (num_envs, 1) — scalar idx avoids extra dim

        # Distance to goal (signed, in world frame)
        self.prev_command_obs = self.command_obs.clone()
        self.command_obs = self.commands[:, 0:1] - self.robotbody_pos_x  # (num_envs, 1)

        # Check if reached goal — 0.05 was too tight, robot oscillates without triggering
        dist_to_goal = torch.abs(self.command_obs).squeeze(-1)  # (num_envs,)
        self.reached_goal = dist_to_goal < 0.10                 # (num_envs,) bool — 10cm threshold

        # Foot end-effector height (z position in world frame) — for swing quality reward
        self.foot_height = self.robot.data.body_link_pose_w[:, self._ee_body_idx, 2]  # (num_envs,)

        # Zero out phase signals when at goal (robot should stand still)
        self.C_frc = torch.where(self.reached_goal, torch.zeros_like(self.C_frc), self.C_frc)
        self.C_vel = torch.where(self.reached_goal, torch.zeros_like(self.C_vel), self.C_vel)

        # ── Apply per-component noise (only during training) ─────────────────
        # Mirrors Manager-Based ObsTerm noise values
        # Uniform noise: (rand * 2 - 1) * range  →  samples in [-range, +range]
        if True: #self.training
            noise_dof_pos = (torch.rand_like(dof_pos) * 2.0 - 1.0) * 0.01   # ±0.01  (joint encoder)
            noise_dof_vel = (torch.rand_like(dof_vel) * 2.0 - 1.0) * 1.5    # ±1.50  (velocity estimator)
            noise_robot_x = (torch.rand_like(self.robot_pos_x) * 2.0 - 1.0) * 0.005  # ±0.005 (linear guide encoder)

            obs_dof_pos = dof_pos + noise_dof_pos
            obs_dof_vel = dof_vel + noise_dof_vel
            obs_robot_x = self.robot_pos_x + noise_robot_x
        else:
            # Eval / play — no noise, use clean sensor values
            obs_dof_pos = dof_pos
            obs_dof_vel = dof_vel
            obs_robot_x = self.robot_pos_x

        # ── Build observation vector ──────────────────────────────────────
        # Total: 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1 = 14
        obs = torch.cat([
            obs_dof_pos,                                    # (num_envs, 3)
            obs_dof_vel,                                    # (num_envs, 3)
            self.prev_actions,                             # (num_envs, 3)
            obs_robot_x,                              # (num_envs, 1)
            self.C_frc.unsqueeze(-1),                      # (num_envs, 1)
            self.C_vel.unsqueeze(-1),                      # (num_envs, 1)
            self.is_foot_in_contact.float().unsqueeze(-1), # (num_envs, 1) — float THEN unsqueeze, not double unsqueeze
            self.command_obs,                               # (num_envs, 1)
        ], dim=-1)

        # ── Extras for logging ────────────────────────────────────────────
        tip_vel = torch.norm(
            self.robot.data.body_lin_vel_w[:, self._ee_body_idx, :3],  # scalar idx → (num_envs, 3)
            dim=-1
        )  # (num_envs,)

        if "obs_extras" not in self.extras:
            self.extras["obs_extras"] = {}
        self.extras["obs_extras"].update({
            "pos_raw": dof_pos.detach(),
            "vel_raw": dof_vel.detach(),
            "actions_clip": self.actions.detach(),
            "prev_actions": self.prev_actions.detach(),
            "robot_pos_x": self.robot_pos_x.detach(),
            "phase_frc": self.C_frc.detach(),
            "phase_vel": self.C_vel.detach(),
            "phase_value": self.phase_value.detach(),
            "torque_raw": self.robot.data.applied_torque[:, self.actuated_dof_indices].detach(),
            "accel_raw": self.robot.data.joint_acc[:, self.actuated_dof_indices].detach(),
            "total_contact_force": self.foot_contact_force.detach(),
            "tip_vel": tip_vel.detach(),
        })

        return {"policy": obs}

    # ══════════════════════════════════════════════════════════════════════
    # Rewards
    # ══════════════════════════════════════════════════════════════════════
    def _get_rewards(self) -> torch.Tensor:
        """
        Reward structure for intermittent one-legged locomotion:

        1. PROGRESS:  Reward forward displacement toward goal
        2. PHASE:     Stance=contact, Swing=foot clearance (NOT raw velocity)
        3. DISTANCE:  Shaped distance-to-goal for gradient signal
        4. STAND:     When at goal, hold default pose
        5. REGULARIZATION: action rate + jerk + torque + acceleration
        """

        # ── Tip velocity and foot height ──────────────────────────────────
        tip_vel = torch.norm(
            self.robot.data.body_lin_vel_w[:, self._ee_body_idx, :3],
            dim=-1
        )  # (num_envs,)

        # ══════════════════════════════════════════════════════════════════
        # 1. PROGRESS REWARD
        # ══════════════════════════════════════════════════════════════════
        prev_dist_to_goal = torch.abs(self.prev_command_obs).squeeze(-1)
        curr_dist_to_goal = torch.abs(self.command_obs).squeeze(-1)
        progress = prev_dist_to_goal - curr_dist_to_goal
        r_progress = torch.clamp(progress * 100.0, -1.0, 2.0)

        # ══════════════════════════════════════════════════════════════════
        # 2. PHASE REWARD — the key fix for jerky swing
        #
        #    OLD: swing reward = tip velocity → policy oscillates joints
        #         rapidly to maximize speed → JERKY
        #
        #    NEW: swing reward = foot clearance (height above ground)
        #         This rewards the foot being LIFTED smoothly during swing,
        #         not just moving fast. Combined with the EMA filter and
        #         rate limiting, this produces a smooth arc.
        #
        #    Stance: reward ground contact (unchanged)
        # ══════════════════════════════════════════════════════════════════
        # Stance: reward contact force
        contact_reward = 1.0 - torch.exp(-0.01 * self.foot_contact_force)  # [0, 1]

        # Swing: reward foot height above a reference ground level
        # foot_height is the z-position of the end effector in world frame
        # We want the foot to lift during swing — reward height above ground
        # Assuming ground is at z ≈ 0, but the foot rests at some z_rest when standing
        # Use a soft reward that saturates — we don't need the foot super high
        foot_clearance = torch.clamp(self.foot_height - 0.02, min=0.0)  # height above 2cm
        swing_quality = 1.0 - torch.exp(-20.0 * foot_clearance)  # saturates around 5cm clearance

        # Also give a small reward for FORWARD tip velocity during swing
        # but much less than before — clearance is the main signal
        tip_vel_x = torch.sign(self.robot.data.body_lin_vel_w[:, self._ee_body_idx, 0])  # x-component only
        # Reward forward velocity in the direction of the goal
        goal_direction = torch.sign(self.command_obs).squeeze(-1)  # +1 or -1
        forward_vel_reward = torch.clamp(tip_vel_x * goal_direction * 0.5, -0.2, 0.5)

        # Combine swing reward: mostly clearance + some forward velocity
        swing_reward = 0.7 * swing_quality + 0.3 * torch.clamp(forward_vel_reward, min=0.0)

        # Phase-gated rewards
        r_phase = (-self.C_frc) * contact_reward + (-self.C_vel) * swing_reward

        # ══════════════════════════════════════════════════════════════════
        # 3. DISTANCE REWARD
        # ══════════════════════════════════════════════════════════════════
        dist = torch.abs(self.command_obs).squeeze(-1)
        r_distance = torch.exp(-2.0 * dist)

        # ══════════════════════════════════════════════════════════════════
        # 4. STAND REWARD — hold default pose when at goal
        # ══════════════════════════════════════════════════════════════════
        joint_error = torch.sum(
            (self.robot.data.joint_pos[:, self.actuated_dof_indices]
             - self.default_joint[:, self.actuated_dof_indices]) ** 2,
            dim=1
        )
        r_stand = torch.where(
            self.reached_goal,
            torch.exp(-5.0 * joint_error),
            torch.zeros_like(joint_error)
        )

        # ══════════════════════════════════════════════════════════════════
        # 5. REGULARIZATION — much stronger than before
        #
        #    Key additions:
        #    - Action rate penalty 5x stronger: -0.01 → -0.05
        #    - NEW jerk penalty: penalizes second derivative of actions
        #      jerk = a(t) - 2*a(t-1) + a(t-2)
        #      This specifically targets high-frequency oscillation
        # ══════════════════════════════════════════════════════════════════
        action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=1)

        # Jerk = second derivative of actions (the "change of change")
        # High jerk = the policy is oscillating, not moving smoothly
        action_jerk = torch.sum(
            (self.actions - 2.0 * self.prev_actions + self.prev_prev_actions) ** 2,
            dim=1
        )

        torques_sq = torch.sum(self.robot.data.applied_torque[:, self.actuated_dof_indices] ** 2, dim=1)
        torque = self.robot.data.applied_torque[:, self.actuated_dof_indices]
        joint_vel = self.robot.data.joint_vel[:, self.actuated_dof_indices]
        accels = torch.sum(self.robot.data.joint_acc[:, self.actuated_dof_indices] ** 2, dim=1)

        energy = torch.sum(torch.abs(torque * joint_vel), dim=1)

        r_regularization = (
            -0.01 * action_rate       # 5x stronger than before
            -0.02 * action_jerk       # NEW: anti-oscillation
            -2.5e-5 * torques_sq
            -6.0e-7 * accels
            -2.5e-5 * energy
        )

        # ══════════════════════════════════════════════════════════════════
        # TOTAL REWARD
        # ══════════════════════════════════════════════════════════════════
        W_PROGRESS = 2.0
        W_PHASE = 1.5
        W_DISTANCE = 1.0
        W_STAND = 1.0       # increased from 0.5 — stronger incentive to stop at goal
        W_REG = 1.0

        # When at goal: mostly reward standing, reduce locomotion rewards
        locomotion_scale = torch.where(
            self.reached_goal,
            torch.tensor(0.1, device=self.device),
            torch.tensor(1.0, device=self.device),
        )

        reward = (
            W_PROGRESS * r_progress * locomotion_scale
            + W_PHASE * r_phase * locomotion_scale
            + W_DISTANCE * r_distance
            + W_STAND * r_stand
            + W_REG * r_regularization
        )

        # ── Logging ───────────────────────────────────────────────────────
        reward_dict = {
            "progress_reward": W_PROGRESS * r_progress,
            "phase_reward": W_PHASE * r_phase,
            "distance_reward": W_DISTANCE * r_distance,
            "stand_reward": W_STAND * r_stand,
            "regularization": W_REG * r_regularization,
            "energy": -2.50e-4 * energy * W_REG,
            "total_reward": reward,
        }

        self.extras["obs_extras"].update({
            "tip_vel": tip_vel.detach(),
            "phase_frc": self.C_frc.detach(),
            "contact_reward": contact_reward.detach(),
            "swing_quality": swing_quality.detach(),
            "foot_clearance": foot_clearance.detach(),
            "action_jerk": action_jerk.detach(),
        })

        for key, value in reward_dict.items():
            self._episode_reward[key] += value

        return reward

    # ══════════════════════════════════════════════════════════════════════
    # Terminations
    # ══════════════════════════════════════════════════════════════════════
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        died = torch.zeros_like(time_out, dtype=torch.bool)

        # ── Track success for curriculum ──────────────────────────────────
        done_envs = time_out | died
        if done_envs.any():
            reached = self.reached_goal[done_envs].float().mean()
            self.success_rate_buf[self.success_idx % 100] = reached
            self.success_idx += 1
            # Advance curriculum if success rate > 70%
            if self.success_idx >= 100:
                avg_success = self.success_rate_buf.mean()
                if avg_success > 0.7 and self.curriculum_level < len(self.curriculum_goals) - 1:
                    self.curriculum_level += 1
                    print(f"[CURRICULUM] Advanced to level {self.curriculum_level}, "
                          f"max_dist={self.curriculum_goals[self.curriculum_level]}")

        # ── Logging ───────────────────────────────────────────────────────
        extras = {}
        for key in self._episode_reward:
            extras[key] = torch.mean(self._episode_reward[key])
            self._episode_reward[key][:] = 0.0
        extras["curriculum_level"] = float(self.curriculum_level)

        self.extras["log"] = extras

        return died, time_out

    # ══════════════════════════════════════════════════════════════════════
    # Reset
    # ══════════════════════════════════════════════════════════════════════
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # ── Reset actions ─────────────────────────────────────────────────
        self.actions[env_ids] = 0.0
        self.actions_raw[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_prev_actions[env_ids] = 0.0

        # ── Reset robot joints ────────────────────────────────────────────
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        # Small perturbation to break symmetry (but not too much!)
        noise = torch.empty(num_reset, self.robot.num_joints, device=self.sim.device).uniform_(-0.05, 0.05)
        joint_pos = torch.clamp(
            joint_pos + noise,
            self.robot.data.joint_pos_limits[env_ids, :, 0],
            self.robot.data.joint_pos_limits[env_ids, :, 1],
        )
        joint_vel = torch.zeros(num_reset, self.robot.num_joints, device=self.sim.device)

        # ── Reset root pose ───────────────────────────────────────────────
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ── Reset phase clock ─────────────────────────────────────────────
        # Cycle time in SECONDS — typical hop cycle is 0.4-0.6s
        self.cycle_time[env_ids] = torch.empty(
            num_reset, dtype=torch.float32, device=self.sim.device
        ).uniform_(0.6, 0.6)
        self.phase_time[env_ids] = 0.0

        # ── Set new goal with curriculum ──────────────────────────────────
        max_dist = self.curriculum_goals[self.curriculum_level]
        # Generate goals in [-max_dist, max_dist] but NOT too close (> 0.2m)
        # raw_targets = torch.empty(num_reset, 1, device=self.sim.device).uniform_(-max_dist, max_dist)
        raw_targets = torch.empty(num_reset, 1, device=self.sim.device).uniform_(-5.0, 5.0)  # for testing full range

        # Ensure minimum distance
        sign = torch.sign(raw_targets)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        raw_targets = torch.where(
            torch.abs(raw_targets) < 0.3,
            sign * 0.3,
            raw_targets,
        )
        self.target_x[env_ids] = raw_targets

        self.commands[env_ids, 0] = self.scene.env_origins[env_ids, 0] + self.target_x[env_ids].squeeze() - 0.07444
        self.commands[env_ids, 1] = self.scene.env_origins[env_ids, 1] - 0.43475
        self.commands[env_ids, 2] = 0.6

        # ── Reset position tracking ───────────────────────────────────────
        self.robot_pos_x[env_ids] = self.robot.data.joint_pos[env_ids][:, self._linear_joint_idx]
        self.prev_robot_pos_x[env_ids] = self.robot_pos_x[env_ids].clone()

        # Reset command_obs so first-step progress reward is zero (no spurious reward)
        body_x = self.robot.data.body_link_pose_w[env_ids, self._linear_body_idx, 0:1]
        self.command_obs[env_ids] = self.commands[env_ids, 0:1] - body_x
        self.prev_command_obs[env_ids] = self.command_obs[env_ids].clone()

        # ── Visualize goal ────────────────────────────────────────────────
        self.goal_visualizer.visualize(self.commands[:])

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════
    def _get_foot_status(self):
        """Binary foot contact status: 1.0 if contact, -1.0 if no contact."""
        f = self.scene["contact_sensor"].data.net_forces_w
        foot_force_norm = torch.norm(f, dim=-1)
        return torch.where(
            foot_force_norm > 1.0,
            torch.ones_like(foot_force_norm),
            -torch.ones_like(foot_force_norm),
        )
    
def sigmoid(x, kappa):
    return 1 / (1 + torch.exp(-kappa * x))