import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from physical_locomotion_ai.tasks.leg_locomotion.one_leg_uan_env_cfg import OneLegUANEnvCfg
from physical_locomotion_ai.uan import UANHardwareDataset


class OneLegUANCalibrationTask(DirectRLEnv):
    """Train a residual actuator model by matching hardware transitions.

    The environment replays real hardware logs. At each step, the simulator is
    commanded with the recorded position target plus the policy's residual
    feed-forward effort. The reward is high when simulated next joint 
    positions/velocities match the recorded next transition.
    """

    cfg: OneLegUANEnvCfg

    def __init__(self, cfg: OneLegUANEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actuated_dof_indices = [self.robot.joint_names.index(name) for name in cfg.actuated_joint_names]
        self.actuated_dof_indices.sort()

        self.dataset = UANHardwareDataset.load(
            cfg.dataset_paths,
            episode_steps=self.max_episode_length,
            device=self.device,
        )

        self.actions = torch.zeros(self.num_envs, cfg.action_space, dtype=torch.float32, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.error_history = torch.zeros(
            self.num_envs,
            cfg.history_len,
            2 * len(self.actuated_dof_indices),
            dtype=torch.float32,
            device=self.device,
        )
        self.data_start_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_data_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for key in ["q_tracking", "dq_tracking", "smoothness", "effort", "total"]
        }

        print(
            "[INFO] UAN dataset loaded: "
            f"{self.dataset.num_rows} rows, {self.dataset.num_valid_starts} valid starts, "
            f"median dt={self.dataset.dt_median:.5f}s"
        )
        for path, length in zip(self.dataset.source_files, self.dataset.source_lengths):
            print(f"[INFO]   {path}: {length} rows")

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.clamp(actions, -1.0, 1.0)
        self.step_data_indices = self.data_start_indices + self.episode_length_buf

    def _apply_action(self):
        command = self.dataset.cmd[self.step_data_indices]
        residual_effort = self.actions * self.cfg.residual_effort_scale

        self.robot.set_joint_position_target(command, joint_ids=self.actuated_dof_indices)
        self.robot.set_joint_effort_target(residual_effort, joint_ids=self.actuated_dof_indices)

    def _get_observations(self) -> dict:
        data_indices = self.data_start_indices + self.episode_length_buf
        real_q = self.dataset.q[data_indices]
        real_dq = self.dataset.dq[data_indices]
        sim_q = self.robot.data.joint_pos[:, self.actuated_dof_indices]
        sim_dq = self.robot.data.joint_vel[:, self.actuated_dof_indices]

        error = torch.cat((real_q - sim_q, real_dq - sim_dq), dim=-1)
        self.error_history = torch.roll(self.error_history, shifts=-1, dims=1)
        self.error_history[:, -1, :] = error
        obs = self.error_history.reshape(self.num_envs, -1)

        if "obs_extras" not in self.extras:
            self.extras["obs_extras"] = {}
        obs_extras = {
            "real_q": real_q.detach(),
            "sim_q": sim_q.detach(),
            "real_dq": real_dq.detach(),
            "sim_dq": sim_dq.detach(),
            "residual_effort": (self.actions * self.cfg.residual_effort_scale).detach(),
            "sim_applied_torque": self.robot.data.applied_torque[:, self.actuated_dof_indices].detach(),
        }
        if self.dataset.tau is not None:
            obs_extras["real_tau"] = self.dataset.tau[data_indices].detach()
        self.extras["obs_extras"].update(obs_extras)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        real_q_next = self.dataset.q_next[self.step_data_indices]
        real_dq_next = self.dataset.dq_next[self.step_data_indices]
        sim_q = self.robot.data.joint_pos[:, self.actuated_dof_indices]
        sim_dq = self.robot.data.joint_vel[:, self.actuated_dof_indices]

        q_error = real_q_next - sim_q
        dq_error = real_dq_next - sim_dq

        q_tracking = torch.exp(-torch.sum(torch.square(q_error / self.cfg.q_error_std), dim=-1))
        dq_tracking = torch.exp(-torch.sum(torch.square(dq_error / self.cfg.dq_error_std), dim=-1))
        smoothness = -torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)
        effort = -torch.sum(torch.square(self.actions), dim=-1)

        reward = (
            self.cfg.q_error_weight * q_tracking
            + self.cfg.dq_error_weight * dq_tracking
            + self.cfg.smoothness_weight * smoothness
            + self.cfg.effort_weight * effort
        )

        terms = {
            "q_tracking": q_tracking,
            "dq_tracking": dq_tracking,
            "smoothness": smoothness,
            "effort": effort,
            "total": reward,
        }
        for key, value in terms.items():
            self._episode_sums[key] += value

        self.prev_actions = self.actions.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        died = torch.zeros_like(time_out, dtype=torch.bool, device=self.device)

        if torch.any(time_out):
            done_ids = time_out.nonzero(as_tuple=False).squeeze(-1)
            self.extras["log"] = {}
            denom = torch.clamp(self.episode_length_buf[done_ids].float(), min=1.0)
            for key, value in self._episode_sums.items():
                self.extras["log"][key] = torch.mean(value[done_ids] / denom)
                value[done_ids] = 0.0

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        num_reset = len(env_ids)
        starts = self.dataset.sample_starts(num_reset, device=self.device)
        self.reset_to_dataset_starts(starts, env_ids)

    def reset_to_dataset_starts(self, starts: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Reset selected environments to explicit hardware dataset rows.

        This is mainly used by evaluation scripts to replay the same hardware
        transitions for different UAN checkpoints.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        starts = starts.to(device=self.device, dtype=torch.long)
        if starts.shape[0] != len(env_ids):
            raise ValueError(f"Expected {len(env_ids)} start indices, got {starts.shape[0]}.")

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self.data_start_indices[env_ids] = starts
        self.step_data_indices[env_ids] = starts

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, self.actuated_dof_indices] = self.dataset.q[starts]
        joint_vel[:, self.actuated_dof_indices] = self.dataset.dq[starts]

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.error_history[env_ids] = 0.0
        for value in self._episode_sums.values():
            value[env_ids] = 0.0
