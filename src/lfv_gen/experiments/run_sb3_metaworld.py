from lfv_gen.experiments.utils import env_utils, log_utils
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.logger import configure
import simple_parsing

algos = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def run_online_expt(algo: str):
    env = env_utils.setup_env(
        "assembly-v2-goal-observable",
        "top_cap2",
    )

    # Default: 2-layer MLP of hidden dims [256, 256]
    model = algos[algo]("MlpPolicy", env, verbose=1)

    wandb_config = log_utils.WandbConfig(
        group="metaworld-online",
        name=f"{algo}-metaworld",
    )
    _logger = log_utils.WandbLogger(wandb_config)
    sb3_logger = configure("/tmp/sb3_log/", ["tensorboard", "stdout", "csv"])

    model.set_logger(sb3_logger)
    model.learn(total_timesteps=10000, log_interval=4)

    metrics = env_utils.eval_policy(
        policy=lambda obs: model.predict(obs, deterministic=True)[0],
        eval_env=env,
        num_episodes=10,
    )
    _logger.log({f"eval/{k}": v for k, v in metrics.items()})
    _logger.finish()


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_argument("--algo", type=str, default="sac")
    args = parser.parse_args()

    if args.algo == "all":
        for algo in algos:
            run_online_expt(algo)
    else:
        run_online_expt(args.algo)
