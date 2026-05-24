
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os
##
# Configuration
##


script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
one_leg_path = os.path.normpath(
    os.path.join(script_dir, "..", "models", "one_leggy8.usd")
)

ONE_LEG_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=one_leg_path,
        activate_contact_sensors=True,  # This is the key addition!
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
            # Add mass properties to fix inertia tensor issues
            # max_linear_velocity=1000.0,
            # max_angular_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        # Add mass properties configuration to fix inertia issues
        # mass_props=sim_utils.MassPropertiesCfg(
        #     mass=1.0,  # Default mass for bodies that don't have proper mass
        # ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.01),
        rot=(0.707, 0.0, 0.0, 0.707),
        joint_pos={
            # ".*": 0.0,
            "linear_left_right": 0.1, 
            # "hip_joint": -1.57,
            # "knee_joint": -0.3926990817 ,
            # "ankle_joint": 0.3926990817 , 
        },
        # 22.5 :0.3926990817 ,
        # 45 :0.7853981634 ,
        # TODO : after change the Z-axis pos in simulation please also change the default joint pos to be the same as the real one

        joint_vel={".*": 0.0},
    ),
    
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint"],
            # stiffness=10.0, #50.0
            # damping=4.75,
            stiffness=50.0, #50.0
            damping=1.0,
            # velocity_limit_sim=1.0472,
            # effort_limit_sim=20
        ),
    },
)

# Define the base link name for your hexapod
ONE_LEG_CFG.base_link_names = ["base_link"]
