import envlogger
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import pathlib

from tqdm import tqdm
from envlogger.backends.tfds_backend_writer import TFDSBackendWriter
from dm_env_wrappers import GymnasiumWrapper
from tensorflow_datasets.rlds import rlds_base
from typing import Literal

from lfv_gen.config import DATASETS_DIR
from lfv_gen.core.types import GymEnv, DmEnv
from lfv_gen.core.policy import Policy
from lfv_gen.envs.wrap_env_as_pixel import wrap_env_as_pixel

ObservationType = Literal["state", "image"]


def collect_and_save_data(
    env: GymEnv,
    policy: Policy,
    num_episodes: int,
    dataset_name: str,
    dataset_dir: pathlib.Path = DATASETS_DIR,
    split_name: str = "train",
    max_episodes_per_file: int = 1000,
    show_progress: bool = False,
):
    # Wrap the env to have pixel observations
    env = wrap_env_as_pixel(env)
    dm_env: DmEnv = GymnasiumWrapper(env)

    # Get the dataset config and save path
    dataset_config = _get_rlds_dataset_config(dm_env, dataset_name=dataset_name)
    dataset_dir = dataset_dir / dataset_name
    dataset_dir.mkdir(exist_ok=True, parents=True)

    # Collect the data
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
                action = policy(timestep.observation["state"])
                timestep = env.step(action)


def _validate_obs_spec(obs_spec: tf.TensorSpec):
    assert set(obs_spec.keys()) == {"pixels", "state"}
    assert len(obs_spec["pixels"].shape) == 3
    assert obs_spec["pixels"].shape[-1] == 3
    assert obs_spec["pixels"].dtype == np.uint8


def _get_rlds_dataset_config(
    dm_env: DmEnv,
    dataset_name: str,
) -> rlds_base.DatasetConfig:
    obs_spec = dm_env.observation_spec()
    _validate_obs_spec(obs_spec)
    act_spec = dm_env.action_spec()

    dataset_config = rlds_base.DatasetConfig(
        name=dataset_name,
        description="Expert data for metaworld environment",
        observation_info=tfds.features.FeaturesDict(
            dict(
                pixels=tfds.features.Image(
                    shape=obs_spec["pixels"].shape,
                    dtype=obs_spec["pixels"].dtype,
                ),
                state=tfds.features.Tensor(
                    shape=obs_spec["state"].shape,
                    dtype=obs_spec["state"].dtype,
                ),
            )
        ),
        action_info=tfds.features.Tensor(
            shape=act_spec.shape,
            dtype=act_spec.dtype,
            encoding=tfds.features.Encoding.ZLIB,
        ),
        reward_info=tf.float64,
        discount_info=tf.float64,
    )
    return dataset_config
