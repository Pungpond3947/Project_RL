import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, RigidBodyMaterialCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from physical_locomotion_ai.assets.models.one_leg import ONE_LEG_CFG


@configclass
class OneLegUANDeployableEnvCfg(DirectRLEnvCfg):
    """Deployable-observation UAN calibration task.

    Unlike the oracle replay UAN, this task does not feed real-sim error to the
    policy. The observation is built from signals available at walking-policy
    runtime: command tracking error, simulated joint state, command change, and
    previous residual action. Real data is still used in the reward during
    training to teach the residual actuator model.
    """

    decimation = 2
    episode_length_s = 10.0
    action_space = 3
    per_step_observation_dim = 15
    history_len = 20
    observation_space = 300  # history_len * per_step_observation_dim
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        use_fabric=True,
        enable_scene_query_support=False,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=16,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            enable_stabilization=True,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    robot: ArticulationCfg = ONE_LEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.7)

    viewer = ViewerCfg(eye=(5.0, 5.0, 3.0))
    actuated_joint_names = ["hip_joint", "knee_joint", "ankle_joint"]

    dataset_paths = (
        "real_data/uan_gaussian_20260505_145826.csv",
        "real_data/uan_sine_20260505_142107.csv",
        "real_data/uan_square_20260505_145243.csv",
        # "real_data/exp2_subsets/p100/uan_exp2_gaussian_20260523_162921.csv",
        # "real_data/exp2_subsets/p100/uan_exp2_sine_20260523_160212.csv",
        # "real_data/exp2_subsets/p100/uan_exp2_square_20260523_161601.csv",
        # "real_data/uan_sine_20260524_152938.csv",
    )
    residual_effort_scale = 2.8

    q_error_weight = 4.0
    dq_error_weight = 0.5
    smoothness_weight = 0.02
    effort_weight = 0.001
    q_error_std = 0.05
    dq_error_std = 0.5
