import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg , ViewerCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg ,PhysxCfg , RigidBodyMaterialCfg, RigidBodyPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg , ContactSensor
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg

import math
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import isaacsim.core.utils.torch as torch_utils

from physical_locomotion_ai.assets.models.one_leg import ONE_LEG_CFG

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )


@configclass
class OneLegEnvCfg(DirectRLEnvCfg):

    decimation = 4 # controlFrequencyInv
    episode_length_s = 20 #30
    action_space = 3
    observation_space = 11
    state_space = 0

    # action_space = gym.spaces.Box(
    #     low=-1.0,
    #     high=1.0,
    #     shape=(6,),
    #     dtype=float
    # )

    # observation_space = gym.spaces.Box(
    #     low=-float("inf"),
    #     high=float("inf"),
    #     shape=(48,),
    #     dtype=float
    # )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        use_fabric = True,
        enable_scene_query_support = False,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=10.0,
            dynamic_friction=10.0,
            restitution=0.0
        ) ,
        physx= PhysxCfg(
            solver_type=1,
            max_position_iteration_count=16,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            enable_stabilization=True,
            ## GPU cap (optional)
            # gpu_max_rigid_contact_count=524288,
            # gpu_max_rigid_patch_count=81920,
            # gpu_found_lost_pairs_capacity=8192,
            # gpu_found_lost_aggregate_pairs_capacity=262144,
            # gpu_total_aggregate_pairs_capacity=8192,
            # gpu_heap_capacity=1048576,
            # gpu_temp_buffer_capacity=1048576,
            # gpu_max_num_partitions=67108864,
            # gpu_max_soft_body_contacts=16777216,
            # gpu_max_particle_contacts=8,
        )
    )

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot : ArticulationCfg = ONE_LEG_CFG.replace(
        prim_path="/World/envs/env_.*/Robot")
    
    contact_debug_vis = False
    contact_force = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/end_effector",
        update_period=0.0,
        history_length=6,
        debug_vis=contact_debug_vis,
        track_air_time = True
        )
    
    # Set View
    viewer = ViewerCfg(eye=(5.0, 5.0, 3.0))
    actuated_joint_names  = ['hip_joint', 'knee_joint' , 'ankle_joint']

    # Set weight
    periodic_weight = 0.1
    timeout_weight  = 2.0
    goal_weight     = 1000.0
    distance_weight = 7.5

    target_threshold = 0.05




@configclass
class OneLegP2PEnvCfg(DirectRLEnvCfg):

    decimation = 4 # controlFrequencyInv
    episode_length_s = 30 #30
    action_space = 3
    observation_space = 13
    state_space = 0

    # action_space = gym.spaces.Box(
    #     low=-1.0,
    #     high=1.0,
    #     shape=(6,),
    #     dtype=float
    # )

    # observation_space = gym.spaces.Box(
    #     low=-float("inf"),
    #     high=float("inf"),
    #     shape=(48,),
    #     dtype=float
    # )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        use_fabric = True,
        enable_scene_query_support = False,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            # static_friction=10.0,
            # dynamic_friction=10.0,
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            restitution=0.0
        ) ,
        physx= PhysxCfg(
            solver_type=1,
            max_position_iteration_count=16,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            enable_stabilization=True,
            ## GPU cap (optional)
            # gpu_max_rigid_contact_count=524288,
            # gpu_max_rigid_patch_count=81920,
            # gpu_found_lost_pairs_capacity=8192,
            # gpu_found_lost_aggregate_pairs_capacity=262144,
            # gpu_total_aggregate_pairs_capacity=8192,
            # gpu_heap_capacity=1048576,
            # gpu_temp_buffer_capacity=1048576,
            # gpu_max_num_partitions=67108864,
            # gpu_max_soft_body_contacts=16777216,
            # gpu_max_particle_contacts=8,
        )
    )

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # friction_combine_mode="average",
            # restitution_combine_mode="average",
            # static_friction=1.0,
            # dynamic_friction=1.0,
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # event
    events: EventCfg = EventCfg()
    
    # robot
    robot : ArticulationCfg = ONE_LEG_CFG.replace(
        prim_path="/World/envs/env_.*/Robot")
    
    contact_debug_vis = True
    contact_force = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/end_effector",
        update_period=0.0,
        history_length=6,
        debug_vis=contact_debug_vis,
        track_air_time = True
        )
    
    # Set View
    viewer = ViewerCfg(eye=(5.0, 5.0, 3.0))
    actuated_joint_names  = ['hip_joint', 'knee_joint' , 'ankle_joint']

    # Set weight
    position_weight = 10.0
    smooth_position_weight = 3.0
    phase_weight = 4.0
    # stand_target_weight = 0.5
    weight_stand_target_scale2 = 0.5

    weight_acceleration_penalty = 2.5e-6
    weight_torque_pealty = 3.0e-5
    weight_action_smoothness_scale = 0.01


    # periodic_weight = 0.1
    # timeout_weight  = 2.0
    # goal_weight     = 1000.0
    # distance_weight = 7.5

    target_threshold = 0.03
        
