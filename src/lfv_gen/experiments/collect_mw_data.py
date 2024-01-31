import envlogger
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm

from envlogger.backends.tfds_backend_writer import TFDSBackendWriter
from dm_env_wrappers import GymnasiumWrapper
from dataclasses import dataclass
from tensorflow_datasets.rlds import rlds_base

from lfv_gen.config import DATASETS_DIR
from lfv_gen.core.types import GymEnv, DmEnv
from lfv_gen.core.policy import Policy
from lfv_gen.envs.setup_metaworld_env import (
    setup_metaworld_env,
    setup_metaworld_policy,
)

from lfv_gen.experiments.utils import data_utils


@dataclass
class ExperimentConfig:
    env_name: str = (
        "drawer-open-v2-goal-observable"  # Metaworld environment for evaluation
    )
    camera_name: str = "top_cap2"  # 'top_cap2', 'left_cap2', 'right_cap2'
    seed: int = 0  # Seed for the environment
    num_episodes: int = 200  # Number of episodes to collect


def collect_data(
    env: DmEnv,
    policy: Policy,
    num_episodes: int,
    dataset_config: rlds_base.DatasetConfig,
    split_name: str = "train",
    max_episodes_per_file: int = 1000,
    show_progress: bool = False,
):
    with envlogger.EnvLogger(
        dm_env,
        backend=TFDSBackendWriter(
            dataset_dir,
            split_name=split_name,
            max_episodes_per_file=max_episodes_per_file,
            ds_config=dataset_config,
        ),
    ) as env:
        for _ in tqdm(range(num_episodes), disable=not show_progress, desc="Episodes"):
            timestep = env.reset()
            while not timestep.last():
                action = policy(timestep.observation)
                timestep = env.step(action)


if __name__ == "__main__":
    env_name = "drawer-open-v2-goal-observable"
    camera_name = "top_cap2"
    seed = 0
    num_episodes = 10

    env: GymEnv = setup_metaworld_env(
        env_name=env_name,
        camera_name=camera_name,
        seed=seed,
    )
    dm_env: DmEnv = GymnasiumWrapper(env)
    policy: Policy = setup_metaworld_policy(env_name)
    dataset_name = data_utils.get_dataset_name("metaworld", env_name, camera_name)
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
        reward_info=tf.float64,
        discount_info=tf.float64,
    )

    collect_data(
        env=dm_env,
        policy=policy,
        num_episodes=num_episodes,
        dataset_config=dataset_config,
        show_progress=True,
    )