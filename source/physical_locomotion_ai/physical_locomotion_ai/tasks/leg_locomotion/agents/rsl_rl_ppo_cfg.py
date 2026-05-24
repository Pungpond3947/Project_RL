from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class OneLegFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000 #1500
    save_interval = 200 #50
    # experiment_name = "one_leg_flat"
    experiment_name = "one_leg_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # actor_hidden_dims=[512, 256, 128],
        # critic_hidden_dims=[512, 256, 128],
        # actor_hidden_dims=[128, 128, 128],
        # critic_hidden_dims=[128, 128, 128],

        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],

        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class OneLegUANPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 100
    experiment_name = "one_leg_uan"
    empirical_normalization = True
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class OneLegUANDeployablePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 100
    experiment_name = "one_leg_uan_deploy"
    empirical_normalization = True
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# @configclass
# class AnymalDFlatPPzORunnerCfg(AnymalDRoughPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         self.max_iterations = 300
#         self.experiment_name = "anymal_d_flat"
#         self.policy.actor_hidden_dims = [128, 128, 128]
#         self.policy.critic_hidden_dims = [128, 128, 128]
