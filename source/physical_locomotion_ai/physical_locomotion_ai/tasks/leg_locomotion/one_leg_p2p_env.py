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

from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
import isaacsim.core.utils.torch as torch_utils

from physical_locomotion_ai.assets.models.one_leg import ONE_LEG_CFG
from physical_locomotion_ai.tasks.leg_locomotion.one_leg_env_cfg import *



class OneLegP2PTask(DirectRLEnv):
    cfg: OneLegEnvCfg

    def __init__(self, cfg: OneLegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(joint_name))
            print(f"Joint name : {joint_name} | Index : {self.robot.joint_names.index(joint_name)}")
        self.actuated_dof_indices.sort()

        # define joint dof index
        self._joint_dof_idx, self._joint_names = self.robot.find_joints(".*")
        print("Hi to all joint!" , self.robot.find_joints(".*"))
        print(f"[DEBUGGING] Joints name: {self._joint_names}") # ['linear_left_right', 'hip_joint', 'knee_joint', 'ankle_joint']
        print(f"[DEBUGGING] Bodies name: {self.robot.body_names}") # ['world', 'linear_left_right_Link', 'linear_up_down_link', 'hip_link', 'knee_link', 'ankle_link', 'end_effector']
        # Check limit
        print(f"[DEBUGGING] DOF LIMIT LOWER : {self.robot.data.soft_joint_pos_limits[0, :, 0]}")
        print(f"[DEBUGGING] DOF LIMIT UPPER : {self.robot.data.soft_joint_pos_limits[0, :, 1]}")
        print(f"[DEBUGGING] DEFUALT DOF : {self.robot.data.default_joint_pos[0, :]}")
        
        # print()
        # self.JOINT_POS_LOWER_LIMIT = self.robot.data.soft_joint_pos_limits[0, self.actuated_dof_indices, 0]
        # self.JOINT_POS_UPPER_LIMIT = self.robot.data.soft_joint_pos_limits[0, self.actuated_dof_indices, 0]
        # Limit position
        self.dof_pos_limits = torch.full((self.num_envs, len(self.actuated_dof_indices)), math.pi,dtype=torch.float32,device=self.sim.device,)
        # Limit vel
        self.dof_vel_limits = torch.full((self.num_envs, len(self.actuated_dof_indices)), math.pi, dtype=torch.float32,device=self.sim.device,)

        # set X target :)

        # set action param
        self.actions = torch.zeros(self.num_envs , 3 , dtype=torch.float32, device=self.sim.device)
        self.prev_actions = torch.zeros(self.num_envs , 3 , dtype=torch.float32, device=self.sim.device)

        # Foot Contact
        self.foot_contact_force = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.is_foot_in_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)
       
        # Set phase
        self.dphase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device) 
        self.phase  = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.C_frc = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.C_vel = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.cycle_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)

        self.un_pred_joint_pos_error = torch.zeros(self.num_envs , self.robot.num_joints , device=self.sim.device)

        self.counter = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device).unsqueeze(-1) # Counter for phase calculation
        # Robot Pos
        self.default_joint  = self.robot.data.default_joint_pos
        self.robot_pos_x    = torch.zeros(self.num_envs , 1 , dtype=torch.float32, device=self.sim.device)

        # self.target_x       = torch.full((self.num_envs , 1) , 2.0 , dtype=torch.float32, device=self.sim.device)
        self.target_x  = torch.zeros(self.num_envs , 1 , dtype=torch.float32, device=self.sim.device)
        self.commands   = torch.zeros(self.num_envs , 3 , dtype=torch.float32, device=self.sim.device)

        # Example noise scales (adjust based on your robot's sensors)
        self.noise_scales = {
            "dof_pos": 0.01,      # 1% noise on normalized positions
            "dof_vel": 0.05,      # 5% noise on velocities
            "robot_pos": 0.005,   # Small noise on global/link position
            "command": 0.01,      # Noise on the target tracking
        }
        
        marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
        marker_cfg.markers["arrow"].scale = (0.2, 0.2, 0.2)
        self.goal_visualizer = VisualizationMarkers(marker_cfg)
        self.goal_visualizer.set_visibility(True)

        # Extras for Log / Analyze
        # Logging
        self._episode_reward = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "phase_reward",
                "timeout_reward",
                "position_tracking_reward",
                "smooth_position_tracking_reward",
                "goal_reward",
                "distance_reward",
                "total_reward",
                "action_smoothness_reward",
                "stand_target",
                "force_penalty",
                "acceleration_penalty",
                "regularization",
                "foot_force_penalty",
                "action_limit_penalty",
            ]
        }

        # ── 1. Initialize self._w ──────────────────────────────────────
        self._w = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "w/phase_scale",
                "w/action_smooth_scale",
                "w/rd_scale",
                "w/rd_scale2",
                "w/action_smooth_scale2",
                "w/stand_target_scale2",
            ]
        }

    def _setup_scene(self):
        # get robot cfg
        self.robot = Articulation(self.cfg.robot)
        # Add contact sensor
        self.contact_sensor = ContactSensor(self.cfg.contact_force)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.light = self.scene.cfg
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.counter += 1
        self.actions = actions.clone()
        # self.actions = torch.clamp(self.actions,  self.robot.data.soft_joint_pos_limits[:, self.actuated_dof_indices, 0],
        #                                           self.robot.data.soft_joint_pos_limits[:, self.actuated_dof_indices, 1])

    def _apply_action(self):
        self.robot.set_joint_position_target(self.actions, joint_ids=self.actuated_dof_indices)

    def _get_observations(self) -> dict:
        net_forces = self.scene["contact_sensor"].data.net_forces_w  # (num_envs, num_bodies, 3)
        force_magnitude = torch.norm(net_forces, dim=-1).squeeze(-1)  # (num_envs,)

        # Light smoothing (0.8 new + 0.2 old) to reduce noise without adding too much lag
        self.foot_contact_force = 0.8 * force_magnitude + 0.2 * self.foot_contact_force
        self.is_foot_in_contact = self.foot_contact_force > 1.0

        foot_status = self._get_foot_status()
        self.prev_actions = self.actions.clone()
        # Get joint position and velocity
        dof_pos, dof_vel = self.robot.data.joint_pos[:,self.actuated_dof_indices], self.robot.data.joint_vel[:,self.actuated_dof_indices]

        
        prev_normalize = math_utils.scale_transform(self.prev_actions.clone(),
                                               -self.dof_pos_limits,
                                               self.dof_pos_limits)
        action_normalized = math_utils.scale_transform(self.actions.clone(),
                                               -self.dof_pos_limits,
                                               self.dof_pos_limits)
        
        dphase = self.episode_length_buf[:, None].squeeze()    / self.cycle_time

        self.phase = torch.sin(2 * torch.pi * dphase+torch.pi) # Phase in range [-1, 1]        

        self.C_frc = torch.where(self.phase > 0, -1, 0.0)
        self.C_vel = torch.where(self.phase >= 0, 0, -1.0)


        self.robot_pos_x = self.robot.data.joint_pos[:,self.robot.find_joints("linear_left_right")[0]] # get position of Linear Left Right
        self.robot_pos_y = self.robot.data.joint_pos[:,self.robot.find_joints("linear_left_right")[0]]
        self.robotbody_pos_x = self.robot.data.body_link_pos_w[:, self.robot.find_bodies("linear_left_right_Link")[0], 0] # get position of Linear Left Right Body
        self.command_obs = self.commands[:, 0].unsqueeze(-1) - self.robotbody_pos_x # Command X relative to end effector X
        reach = torch.norm(self.command_obs, dim=1).squeeze(-1) < 0.03
        self.C_frc = torch.where(reach, torch.zeros_like(self.C_frc), self.C_frc)
        self.C_vel = torch.where(reach, torch.zeros_like(self.C_vel), self.C_vel)
        self.phase = torch.where(reach, torch.zeros_like(self.phase), self.phase)

        
        add_noise = True
        if add_noise:
            dof_pos += torch.randn_like(dof_pos) * self.noise_scales["dof_pos"]
            dof_vel += torch.randn_like(dof_vel) * self.noise_scales["dof_vel"]
            self.robot_pos_x += torch.randn_like(self.robot_pos_x) * self.noise_scales["robot_pos"]
            self.command_obs += torch.randn_like(self.command_obs) * self.noise_scales["command"]

         # Normalize Posself.target_x
        pos_scale = math_utils.scale_transform(dof_pos , 
                                               -self.dof_pos_limits,
                                               self.dof_pos_limits)
        # Normalize Vel
        vel_scale = math_utils.scale_transform(dof_vel , 
                                               -self.dof_vel_limits,
                                               self.dof_vel_limits)
        
        obs = torch.cat(
            (
                # dof_pos,
                # dof_vel,
                pos_scale,
                vel_scale,
                # self.actions,
                self.prev_actions,
                # action_normalized,
                # prev_normalize,
                self.robot_pos_x,
                self.is_foot_in_contact.float().unsqueeze(-1),
                self.C_frc.unsqueeze(-1),
                # self.C_vel.unsqueeze(-1),
                # self.phase.unsqueeze(-1),
                self.command_obs, # Add Command to observation
            ),
            dim=-1
        )
        observations = {"policy" : obs}

        # print(f"Obs : {obs}")

        if "obs_extras" not in self.extras:
            self.extras["obs_extras"] = {}

        self.extras["obs_extras"].update({
            # "pos_scale": pos_scale.detach(),
            # "vel_scale": vel_scale.detach(),
            "pos_raw": dof_pos.detach(),
            "vel_raw": dof_vel.detach(),
            "actions_clip": self.actions.detach(),
            'torque_raw': self.robot.data.applied_torque[:,self.actuated_dof_indices].detach(),
            "prev_actions": self.prev_actions.detach(),
            "robot_pos_x": self.robot_pos_x.detach(),
            "phase": self.phase.detach(),
            "episode_length": self.episode_length_buf.detach(),
            "total_contact_force": self.foot_contact_force.detach(),
            "phase_frc" : self.C_frc.detach(),
        })
        # print(f"Obs : {obs}")
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # --- Phase Reward
        # Contact force calculation with history (max force over history)
        net_contact_forces = self.scene["contact_sensor"].data.net_forces_w  # shape (num_envs, 1, 3)
        contact_forces = torch.norm(net_contact_forces, dim=-1).squeeze(-1)            # (num_envs, 1)
        # contact_forces = torch.max(torch.norm(net_contact_forces[:, :, :], dim=-1), dim=1)[0]  # Shape: (num_envs, num_feet)
        total_contact_force = contact_forces  # Shape: (num_envs,) #2
        tip_vel = torch.norm(self.robot.data.body_lin_vel_w[:, self.robot.find_bodies(["end_effector"])[0] , [0,1,2]],dim=-1).squeeze(-1) #3         # Velocity magnitudes

        body_vel = self.robot.data.body_lin_vel_w[:, self.robot.find_bodies(["linear_left_right_Link"])[0] , 0].squeeze(-1)

        # Normalized tanh: maps body_vel through (tanh(x-2)+1)/2
        body_vel_norm = (torch.tanh(torch.abs(body_vel) - 2.0) + 1.0) / 2.0
        phase_reward = (self.C_frc * (1 - torch.exp(-contact_forces)) + self.C_vel * (1 - torch.exp(-tip_vel))) #- body_vel_norm

        # --- Distance Reward
        dist = torch.norm(self.command_obs, dim=1).squeeze(-1) # Distance to target in X direction
        std = 2.0
        position_tracking_reward = 1 - torch.tanh(dist / std)
        std2 = 0.5 # 0.5
        smooth_position_tracking_reward = 1 - torch.tanh(dist / std2)

        reach_condition = torch.norm(self.command_obs, dim=1).squeeze(-1) < 0.03

        stand_target = torch.where(reach_condition, 
                                   torch.exp(-torch.sum((self.robot.data.joint_pos[:, self.actuated_dof_indices] - self.default_joint[:,self.actuated_dof_indices])**2, dim=1)),
                                   torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device))  # reward

        # --- Action Smoothness Penalty
        # action_smoothness = torch.exp(-torch.sum(torch.square(self.actions - self.prev_actions),dim=-1))-1
        action_smoothness = -torch.sum(torch.square(self.actions - self.prev_actions), dim=1)
        weight_action_smoothness_scale = 0.005

        # joint torques
        joint_torques = -torch.sum(torch.square(self.robot.data.applied_torque[:, self.actuated_dof_indices]), dim=1)
        weight_torque_pealty = 3.0e-5
        # joint acceleration
        joint_accel = -torch.sum(torch.square(self.robot.data.joint_acc[:, self.actuated_dof_indices]), dim=1)
        weight_acceleration_penalty = 2.5e-7 # 2.5e-7
        # joint velocity
        # joint_velo = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)

        
        

        # --- Force Penalty
        # print(f"Contact Forces: {contact_forces}")
        foot_force_penalty = -torch.sum(torch.square(torch.maximum(contact_forces.unsqueeze(-1) - 100.0, torch.zeros_like(contact_forces.unsqueeze(-1)))), dim=1)
        weight_foot_force_penalty = 1.0e-7

        action_limit = torch.clamp(self.actions,  self.robot.data.soft_joint_pos_limits[:, self.actuated_dof_indices, 0],
                                                  self.robot.data.soft_joint_pos_limits[:, self.actuated_dof_indices, 1])

        action_limit_penalty = -torch.sum(torch.square(self.actions - action_limit), dim=1)
        weight_action_limit_penalty = 1.0e-3 # TODO : Increase this penalty to see if it helps with action overshooting

        regularization = weight_torque_pealty * joint_torques + weight_acceleration_penalty * joint_accel + weight_foot_force_penalty * foot_force_penalty # + weight_action_limit_penalty * action_limit_penalty # + weight_foot_force_penalty * foot_force_penalty
        # --- Acceleration Penalty
        # acceleration_penalty = torch.exp(-torch.sum(torch.square(body_vel),dim=-1))-1



        # =====================================================================================================================
        # ----------------------------------------- Parallel - ROGER Start-----------------------------------------------------
        # =====================================================================================================================

        phase_boundary_max = 0.0 # 0.5
        phase_boundary_size = 0.3 #1.0
        phase_w = (phase_boundary_max - phase_reward)/phase_boundary_size 
        phase_w_clip = phase_w.clamp(0.0, 1.0) # Clip to [0, 1]
        weight_phase = phase_w_clip

        action_smoothness_boundary_max = torch.deg2rad(torch.tensor(1.0)) # 0.1 degree in radians
        action_smoothness_boundary_size = 2.0
        action_smoothness_w = (action_smoothness_boundary_max - action_smoothness)/action_smoothness_boundary_size # Normalize to [0, 1]
        action_smoothness_w_clip = action_smoothness_w.clamp(0.0, 1.0)# Clip to
        weight_action_smoothness = action_smoothness_w_clip

        sum_weight =  weight_phase + 1e-6 # weight_rough_distance + weight_smooth_distance + weight_stand_target + weight_action_smoothness + 1e-6

        weight_phase_scale = torch.where(sum_weight >= 1.0, weight_phase / sum_weight, weight_phase)
        # weight_action_smoothness_scale = torch.where(sum_weight >= 1.0, weight_action_smoothness / sum_weight, weight_action_smoothness)
        weight_rd_scale = torch.where(sum_weight >= 1.0, 0.0, 1.0 - weight_phase)

        # weight_stand_target_scale = weight_smooth_distance / sum_weight
        reward = (position_tracking_reward  + smooth_position_tracking_reward) * weight_rd_scale + phase_reward * weight_phase_scale + weight_action_smoothness_scale * action_smoothness + regularization ## + force_penalty * 0.1 * weight_phase_scale + acceleration_penalty * 0.1 * weight_phase_scale

        weight_stand_target_scale2 = 0.1 # 0.2
        reward_reach =  (position_tracking_reward  + smooth_position_tracking_reward) * weight_rd_scale + stand_target * weight_stand_target_scale2 + weight_action_smoothness_scale * action_smoothness + regularization


        # =====================================================================================================================
        # ----------------------------------------- Parallel - ROGER End ------------------------------------------------------
        # =====================================================================================================================

        # # --- Condition on reach_condition ---
        reward = torch.where(reach_condition, reward_reach, reward)

        reward_dict = {
            "phase_reward": phase_reward,
            "position_tracking_reward": position_tracking_reward,
            "smooth_position_tracking_reward": smooth_position_tracking_reward,
            "action_smoothness_reward": action_smoothness,
            "stand_target": stand_target,
            "regularization": regularization,
            "foot_force_penalty": foot_force_penalty,
            "action_limit_penalty": action_limit_penalty,
            # "timeout_reward": timeout_reward,
            # "goal_reward": goal_reward,
            # "distance_reward": distace_reward,
            # "force_penalty": force_penalty,
            # "acceleration_penalty": acceleration_penalty,
            "total_reward": reward,
        }

        # ── 2. Build weight_dict each step ────────────────────────────────────────
        weight_dict = {
            "w/phase_scale":          weight_phase_scale.mean().detach(),
            # "w/action_smooth_scale":  weight_action_smoothness_scale.mean().detach(),
            "w/rd_scale":             weight_rd_scale.mean(), #.detach(),
            # "w/rd_scale2":            weight_rd_scale2,
            # "w/action_smooth_scale2": weight_action_smoothness_scale2.mean().detach(),
            "w/stand_target_scale2":  weight_stand_target_scale2,
        }


        # Logging and plot
        reward_dict_log = {
            # "total_contact_force": total_contact_force.detach(),
            "tip_vel": tip_vel.detach(),
            "phase force": self.C_frc.detach(),
            "Phase Force reward": (self.C_frc * total_contact_force).detach(),
            "Phase Velocity reward": (self.C_vel * tip_vel).detach(),
            # "weight_phase_scale": weight_phase_scale.detach(),
            # "weight_stand_target_scale": weight_stand_target_scale.detach(),
            "stand_target": stand_target.detach(),
        }
        self.extras["obs_extras"].update(reward_dict_log)

        # ── 3. Accumulate weights ─────────────────────────────────────────────────
        for key, value in reward_dict.items():
            self._episode_reward[key] += value

        for key, value in weight_dict.items():          # ← separate accumulation
            self._w[key] = value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        died = torch.zeros_like(time_out, dtype=torch.bool, device=self.sim.device)
        # Compute distance
        # dist = torch.norm(self.robot_pos_x - self.target_x, dim=1)
        # died = dist < self.cfg.target_threshold
        
        # Logging Reward
        extras = dict() # Temporary Buffer
        for key in self._episode_reward.keys():
            episodic_sum_avg = torch.mean(self._episode_reward[key])
            extras[key] = episodic_sum_avg #/ self.max_episode_length
            self._episode_reward[key][:] = 0.0

        # weight logging from self._w
        for key in self._w.keys():
            extras[key] = self._w[key]
            # self._w[key][:] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        return died , time_out 
        # return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        # ------------------ Reset Action ------------------ #
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        # ------------------ Reset Robot ------------------ # 
        joint_pos = torch.empty(num_reset , self.robot.num_joints , device=self.sim.device).uniform_(-0.1,0.1)
        joint_vel = torch.empty(num_reset , self.robot.num_joints , device=self.sim.device).uniform_(-0.0,0.0)

        joint_pos[:] = torch.clamp(self.robot.data.default_joint_pos[env_ids] + joint_pos , self.robot.data.joint_pos_limits[env_ids , : , 0], self.robot.data.joint_pos_limits[env_ids , : , 1])

        # # Get Root Pose
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # # Set Root Pose/Vel
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # Set Joint State
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ------------------ Set unpredictable pos joint error to robot ------------------ #
        self.un_pred_joint_pos_error[env_ids] = torch.empty(num_reset , self.robot.num_joints , device=self.sim.device).uniform_(-0.05,0.05)

        # ------------------ Set new cycle time ------------------ # 
        # self.cycle_time[env_ids] = torch.empty(num_reset, dtype=torch.float32, device=self.sim.device).uniform_(100.0, 150.0)
        self.cycle_time[env_ids] = torch.empty(num_reset, dtype=torch.float32, device=self.sim.device).uniform_(150.0, 150.0)


         # ------------------ Set new target ------------------ # 
        self.target_x[env_ids] = torch.empty(num_reset , 1 , device=self.sim.device).uniform_(-3.0, 3.0) # Target X in range [1.0, 2.0]
        self.commands[env_ids,0] = self.scene.env_origins[env_ids,0] + self.target_x[env_ids].squeeze() - 0.07444 #Command X
        self.commands[env_ids,1] = self.scene.env_origins[env_ids,1]- 0.43475 # Command Y
        self.commands[env_ids,2] = 0.6

        # ------------------ Reset Counter ------------------ # 
        self.counter[env_ids] = 0    

        # ------------------ Goal visualizer ------------------ # 
        self.goal_visualizer.visualize(self.commands[:])


    def _get_foot_status(self):
        f = self.scene["contact_sensor"].data.net_forces_w  # shape (num_envs, 4, 3)
        # compute per-foot norm
        foot_force_norm = torch.norm(f, dim=-1)            # (num_envs, 4)
        # threshold at 1.0
        foot_status = torch.where(
            foot_force_norm > 1.0,
            torch.tensor(1.0, device=foot_force_norm.device),
            torch.tensor(-1.0, device=foot_force_norm.device),
        )
        return foot_status

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))
