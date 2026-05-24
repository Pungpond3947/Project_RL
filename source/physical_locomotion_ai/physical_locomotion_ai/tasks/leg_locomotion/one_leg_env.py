import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensor

import isaaclab.utils.math as math_utils

import math
import torch
import numpy as np

from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
import isaacsim.core.utils.torch as torch_utils

from physical_locomotion_ai.assets.models.one_leg import ONE_LEG_CFG
from physical_locomotion_ai.tasks.leg_locomotion.one_leg_env_cfg import *

class OneLegReachGoalTask(DirectRLEnv):
    cfg: OneLegEnvCfg

    def __init__(self, cfg: OneLegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(joint_name))
            print(f"Joint name : {joint_name} | Index : {self.robot.joint_names.index(joint_name)}")
        self.actuated_dof_indices.sort()
        # print(self.actuated_dof_indices.sort())


        # define joint dof index
        self._joint_dof_idx, self._joint_names = self.robot.find_joints(".*")
        print("Hi to all joint!" , self.robot.find_joints(".*"))
        print(f"[DEBUGGING] Joints name: {self._joint_names}") # ['linear_left_right', 'hip_joint', 'knee_joint', 'ankle_joint']
        print(f"[DEBUGGING] Bodies name: {self.robot.body_names}") # ['world', 'linear_left_right_Link', 'linear_up_down_link', 'hip_link', 'knee_link', 'ankle_link', 'end_effector']
        # Check limit
        print(f"[DEBUGGING] DOF LIMIT LOWER : {self.robot.data.soft_joint_pos_limits[0, :, 0]}")
        print(f"[DEBUGGING] DOF LIMIT UPPER : {self.robot.data.soft_joint_pos_limits[0, :, 1]}")
        # print()
        # self.JOINT_POS_LOWER_LIMIT = self.robot.data.soft_joint_pos_limits[0, self.actuated_dof_indices, 0]
        # self.JOINT_POS_UPPER_LIMIT = self.robot.data.soft_joint_pos_limits[0, self.actuated_dof_indices, 0]
        # Limit vel
        self.dof_vel_limits = torch.full((self.num_envs, len(self.actuated_dof_indices)), 1.0472,dtype=torch.float32,device=self.sim.device,)

        # set X target :)
        # self.targets = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))

        # set action param
        self.actions = torch.zeros(self.num_envs , 3 , dtype=torch.float32, device=self.sim.device)
        self.prev_actions = torch.zeros(self.num_envs , 3 , dtype=torch.float32, device=self.sim.device)

        # Foot Contact
        self.foot_contact_force = torch.zeros(self.num_envs , 4 , dtype=torch.float32, device=self.sim.device)
        self.foot_contact_status = torch.zeros(self.num_envs , 4 , dtype=torch.float32, device=self.sim.device)
       
        # Set phase
        self.dphase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device) 
        self.phase  = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        # self.cycle_time = 300 # Higher cyle -> slower Phase
        self.cycle_time = 50 # Higher cyle -> slower Phase

        self.counter = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device).unsqueeze(-1) # Counter for phase calculation
        # Robot Pos
        self.default_joint  = self.robot.data.default_joint_pos
        self.robot_pos_x    = torch.zeros(self.num_envs , 1 , dtype=torch.float32, device=self.sim.device)
        self.target_x       = torch.full((self.num_envs , 1) , 2.0 , dtype=torch.float32, device=self.sim.device)

        # Extras for Log / Analyze
        # Logging
        self._episode_reward = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "phase_reward",
                "timeout_reward",
                "goal_reward",
                "distance_reward",
                "total_reward",
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

    def _apply_action(self):
        self.robot.set_joint_position_target(self.actions, joint_ids=self.actuated_dof_indices)

    def _get_observations(self) -> dict:
        foot_status = self._get_foot_status()
        # Get joint position and velocity
        dof_pos, dof_vel = self.robot.data.joint_pos[:,self.actuated_dof_indices], self.robot.data.joint_vel[:,self.actuated_dof_indices]
        # Normalize Posself.target_x
        pos_scale = math_utils.scale_transform(dof_pos , 
                                               self.robot.data.soft_joint_pos_limits[0, self.actuated_dof_indices, 0],
                                               self.robot.data.soft_joint_pos_limits[0, self.actuated_dof_indices, 1],)
        # Normalize Vel
        vel_scale = math_utils.scale_transform(dof_vel , 
                                               -self.dof_vel_limits,
                                               self.dof_vel_limits)
        self.prev_actions = self.actions.clone()

        dphase = self.episode_length_buf[:, None] / self.cycle_time
        # dphase = self.counter / self.cycle_time

        self.phase = torch.sin(2 * torch.pi * dphase+torch.pi) # Phase in range [-1, 1]
    
        self.robot_pos_x = self.robot.data.joint_pos[:,self.robot.find_joints("linear_left_right")[0]] # get position of Linear Left Right
        # print(f"Robot Pos X : {self.robot_pos_x[0]} | Target X : {self.target_x[0]}")
        # print(self.phase.shape , self.counter.shape , dphase.shape)
        obs = torch.cat(
            (
                pos_scale,
                vel_scale,
                self.prev_actions,
                # foot_status,
                self.robot_pos_x,
                self.phase,
                # Command :(
            ),
            dim=-1
        )
        observations = {"policy" : obs}

        if "obs_extras" not in self.extras:
            self.extras["obs_extras"] = {}

        self.extras["obs_extras"].update({
            "pos_scale": pos_scale.detach(),
            "vel_scale": vel_scale.detach(),
            "prev_actions": self.prev_actions.detach(),
            "robot_pos_x": self.robot_pos_x.detach(),
            "phase": self.phase.detach(),
            "episode_length": self.episode_length_buf.detach(),
        })
        # print(f"Obs : {obs}")
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # --- Phase Reward

        # Contact force calculation with history (max force over history)
        # net_contact_forces = self.scene["contact_sensor"].data.net_forces_w_history
        net_contact_forces = self.scene["contact_sensor"].data.net_forces_w  # shape (num_envs, 1, 3)
        # print(f"Net Contact Forces : {net_contact_forces.shape} | {net_contact_forces[0]}")
        contact_forces = torch.norm(net_contact_forces, dim=-1).squeeze(-1)            # (num_envs, 1)
        # print(f"Contact Forces : {contact_forces.shape} | {contact_forces[0]}")

        # contact_forces = torch.max(torch.norm(net_contact_forces[:, :, :], dim=-1), dim=1)[0]  # Shape: (num_envs, num_feet)
        total_contact_force = contact_forces  # Shape: (num_envs,) #2
        # Velocity magnitudes
        tip_vel = torch.norm(self.robot.data.body_lin_vel_w[:, self.robot.find_bodies(["end_effector"])[0] , [0,1,2]],dim=-1).squeeze(-1) #3

        body_vel = self.robot.data.body_lin_vel_w[:, self.robot.find_bodies(["linear_left_right_Link"])[0] , 0].squeeze(-1) #4
        C_frc = torch.where(self.phase.squeeze() > 0, -1, 0.0)
        C_vel = torch.where(self.phase.squeeze() >= 0, 0, -1.0)

        phase_reward = self.cfg.periodic_weight * (C_frc * contact_forces + C_vel * tip_vel)

        # --- Goal reward
        dist = torch.norm(self.robot_pos_x - self.target_x, dim=1)
        reach = dist < self.cfg.target_threshold
        goal_reward = self.cfg.goal_weight*reach
        # --- Distance reward
        distace_reward = -self.cfg.distance_weight*dist.squeeze(-1)

        reward = phase_reward  + distace_reward


        # print(torch.mean(self.robot.data.root_pos_w[:,2]))
        reward_dict = {
            "phase_reward": phase_reward,
            # "timeout_reward": timeout_reward,
            # "goal_reward": goal_reward,
            "distance_reward": distace_reward,
            "total_reward": reward,
        }


        # Logging and plot
        reward_dict_log = {
            # "total_contact_force": total_contact_force.detach(),
            "tip_vel": tip_vel.detach(),
            "body_vel": body_vel.detach(),
            "phase force": C_frc.detach(),
            "Phase Force reward": (C_frc * total_contact_force).detach(),
            "Phase Velocity reward": (C_vel * tip_vel).detach(),
        }
        self.extras["obs_extras"].update(reward_dict_log)


        for key, value in reward_dict.items():
            self._episode_reward[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)


        died = torch.zeros_like(time_out, dtype=torch.bool, device=self.sim.device)
        # Compute distance
        dist = torch.norm(self.robot_pos_x - self.target_x, dim=1)
        died = dist < self.cfg.target_threshold
        
        # Logging Reward
        extras = dict() # Temporary Buffer
        for key in self._episode_reward.keys():
            episodic_sum_avg = torch.mean(self._episode_reward[key])
            extras[key] = episodic_sum_avg #/ self.max_episode_length
            self._episode_reward[key][:] = 0.0
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
        joint_pos = torch.empty(num_reset , self.robot.num_joints , device=self.sim.device).uniform_(-0.02,0.02)
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

        # ------------------ Reset Counter ------------------ # 
        self.counter[env_ids] = 0    


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