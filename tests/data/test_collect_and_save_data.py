import gymnasium
import tempfile
import pathlib
import tensorflow_datasets as tfds
import tensorflow as tf

from lfv_gen.data import rlds_utils
from lfv_gen.data.collect_and_save_data import collect_and_save_data


def test_collect_and_save_data():
    env = gymnasium.make("Pendulum-v1", render_mode="rgb_array")

    def policy(obs):
        del obs
        return env.action_space.sample()

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_dir = pathlib.Path(tmp_dir)
        dataset_name = "test_dataset"
        split_name = "train"

        # Create the dataset
        collect_and_save_data(
            env=env,
            policy=policy,
            num_episodes=1,
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            split_name=split_name,
            max_episodes_per_file=1000,
            show_progress=False,
        )

        # Test loading the dataset
        builder = tfds.builder_from_directory(dataset_dir.absolute() / dataset_name)
        dataset = builder.as_dataset()[split_name]
        dataset = dataset.map(
            rlds_utils.skip_truncated_last_step, num_parallel_calls=tf.data.AUTOTUNE
        )
        transitions = rlds_utils.episodes_to_transitions_dataset(
            dataset,
            cycle_length=16,
            block_length=16,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        # Test the dataset
        trans = transitions.take(1).as_numpy_iterator().next()
        assert len(trans.observation["pixels"].shape) == 3
        assert len(trans.observation["state"].shape) == 1
