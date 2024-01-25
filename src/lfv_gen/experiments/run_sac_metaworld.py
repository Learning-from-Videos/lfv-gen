from wandb.integration.sb3 import WandbCallback

from lfv_gen.experiments.utils import env_utils, log_utils
from stable_baselines3 import SAC

if __name__ == "__main__":
    env = env_utils.setup_env(
        "assembly-v2-goal-observable",
        "top_cap2",
    )

    # Default: 2-layer MLP of hidden dims [256, 256]
    model = SAC("MlpPolicy", env, verbose=1)

    wandb_config = log_utils.WandbConfig(
        group="default",
        name="sac-metaworld",
    )
    logger = log_utils.WandbLogger(wandb_config)

    model.learn(total_timesteps=10000, log_interval=4, callback=WandbCallback())

    metrics = env_utils.eval_policy(
        policy=lambda obs: model.predict(obs, deterministic=True)[0],
        eval_env=env,
        num_episodes=10,
    )
    logger.log({f"eval/{k}": v for k, v in metrics.items()})
