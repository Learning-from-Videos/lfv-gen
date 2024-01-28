import envlogger
import tensorflow_datasets as tfds

from envlogger.backends.tfds_backend_writer import TFDSBackendWriter
from dm_env_wrappers import GymnasiumWrapper
from dataclasses import dataclass
from tensorflow_datasets.rlds import rlds_base

from lfv_gen.config import DATASETS_DIR
from lfv_gen.core.types import GymEnv, DmEnv
from lfv_gen.core.policy import Policy
from lfv_gen.envs.setup_metaworld_env import (
    MetaworldEnvName,
    MetaworldCameraName,
    setup_metaworld_env,
    setup_metaworld_policy,
)


@dataclass
class ExperimentConfig:
    env_name: str = (
        "drawer-open-v2-goal-observable"  # Metaworld environment for evaluation
    )
    camera_name: str = "top_cap2"  # 'top_cap2', 'left_cap2', 'right_cap2'
    seed: int = 0  # Seed for the environment
    num_episodes: int = 200  # Number of episodes to collect


def get_dataset_name(
    suite: str,
    env_name: MetaworldEnvName,
    camera_name: MetaworldCameraName,
):
    return f"{suite}/{env_name}/{camera_name}"


def collect_data(
    env_name: MetaworldEnvName,
    camera_name: MetaworldCameraName,
    seed: int,
    num_episodes: int,
):
    env: GymEnv = setup_metaworld_env(
        env_name=env_name,
        camera_name=camera_name,
        seed=seed,
    )
    dm_env: DmEnv = GymnasiumWrapper(env)
    policy: Policy = setup_metaworld_policy(env_name)

    dataset_name = get_dataset_name("metaworld", env_name, camera_name)
    dataset_dir = DATASETS_DIR / dataset_name
    dataset_dir.mkdir(exist_ok=True, parents=True)

    # state dataset
    dataset_config = rlds_base.DatasetConfig(
        name=dataset_name,
        description="Expert data for metaworld environment",
        observation_info=tfds.features.Tensor(
            shape=dm_env.observation_spec().shape,
            dtype=dm_env.observation_spec().dtype,
            encoding=tfds.features.Encoding.ZLIB,
        ),
        action_info=tfds.features.Tensor(
            shape=dm_env.action_spec().shape,
            dtype=dm_env.action_spec().dtype,
            encoding=tfds.features.Encoding.ZLIB,
        ),
        reward_info=tfds.features.Tensor(
            shape=dm_env.reward_spec().shape,
            dtype=dm_env.reward_spec().dtype,
            encoding=tfds.features.Encoding.ZLIB,
        ),
        # TODO: figure out how to encode discount
        # discount_info=tf.float32
    )

    with envlogger.EnvLogger(
        dm_env,
        backend=TFDSBackendWriter(
            dataset_dir,
            split_name="train",
            max_episodes_per_file=100,
            ds_config=dataset_config,
        ),
    ) as env:
        for _ in range(num_episodes):
            timestep = env.reset()
            while not timestep.last():
                action = policy(timestep.observation)
                timestep = env.step(action)


if __name__ == "__main__":
    config = ExperimentConfig()
    collect_data(
        env_name=config.env_name,
        camera_name=config.camera_name,
        seed=config.seed,
        num_episodes=config.num_episodes,
    )
