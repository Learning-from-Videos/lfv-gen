from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from lfv_gen.config import ROOT_DIR

import wandb
import simple_parsing
import gymnasium
import tensorflow as tf
import pickle
import numpy as np
from typing import Any
from moviepy.editor import VideoClip

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from jam.haiku import r3m
from jam import imagenet_util
from safetensors.flax import load_file
from tqdm import tqdm
from dataclasses import dataclass

import logging
import os
import shutil
import pathlib

JAM_MODEL_DIR = os.getenv("JAM_MODEL_DIR")

# common types
Environment = gymnasium.Env
Dataset = tf.data.Dataset


@dataclass
class ExperimentConfig:
    dataset_env_name: str = (
        "drawer-open-v2-goal-observable"  # Metaworld environment for dataset
    )
    dataset_env_viewpoint: str = "top_cap2"  # 'top_cap2', 'left_cap2', 'right_cap2'
    dataset_n_episodes: int = 200  # max 200
    eval_env_name: str = (
        "drawer-open-v2-goal-observable"  # Metaworld environment for evaluation
    )
    eval_env_viewpoint: str = "top_cap2"  # 'top_cap2', 'left_cap2', 'right_cap2'
    enc_model_name: str = "r3m/r3m-18"  # 'r3m/r3m-18', 'r3m/r3m-34', 'r3m/r3m-50'
    train_steps: int = 10000  # Training steps
    eval_freq: int = 2500  # Evaluation frequency
    batch_size: int = 256
    learning_rate: float = 3e-4
    seed: int = 42  # Random seed used to initialize policy


@dataclass
class WandbConfig:
    project: str | None = None
    entity: str | None = None
    group: str | None = "default"
    name: str | None = "r3m-bc"


def rm_tree(path: pathlib.Path):
    """Recursively remove a directory and all its contents."""
    shutil.rmtree(path)


def make_video(image_array: np.ndarray, save_path: str, fps: int):
    """
    Convert a numpy array of images into an MP4 video.

    Args:
        image_array (np.ndarray): Numpy array of images with shape (batch, height, width, 3).
        save_path (str): Path where the output MP4 video will be saved.
        fps (int): Frames per second for the output video.

    Returns:
        None
    """

    def make_frame(t):
        frame_idx = int(np.round(t * fps))
        # Clamp frame_idx within valid range
        frame_idx = max(0, min(frame_idx, len(image_array) - 1))
        return image_array[frame_idx, ...]

    video = VideoClip(make_frame, duration=image_array.shape[0] / fps)
    video.write_videofile(save_path, fps=fps, verbose=False)


def run_offline_experiment(config: ExperimentConfig, wandb_config: WandbConfig):
    prng_key = jax.random.PRNGKey(config.seed)

    def setup_eval_env(eval_env_name: str, eval_env_viewpoint: str) -> Environment:
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[eval_env_name]
        e = env_cls(render_mode="rgb_array")

        # Hack: enable random reset
        e._freeze_rand_vec = False

        # Hack: set the camera view  to be the same as R3M
        e.camera_name = eval_env_viewpoint
        # Hack: set the renderer to return 256x256 images
        e.model.vis.global_.offwidth = 256
        e.model.vis.global_.offheight = 256

        # Simple validation
        e.reset()
        image_array = e.render()
        assert image_array.shape == (256, 256, 3)
        return e

    def setup_dataset(dataset_env_name: str, dataset_env_viewpoint: str) -> Dataset:
        dataset_path = (
            f"datasets/metaworld/{dataset_env_viewpoint}/{dataset_env_name}.pickle"
        )
        with open(ROOT_DIR / dataset_path, "rb") as f:
            dataset: list[dict[str, Any]] = pickle.load(f)

        # Simple validation
        assert "images" in dataset[0].keys()
        assert "actions" in dataset[0].keys()
        assert "observations" in dataset[0].keys()
        assert "rewards" in dataset[0].keys()
        assert dataset[0]["images"].shape == (500, 256, 256, 3)

        return dataset

    # Load environment
    logging.info("Setting up environment")
    env: Environment = setup_eval_env(
        eval_env_name=config.eval_env_name, eval_env_viewpoint=config.eval_env_viewpoint
    )

    # Load dataset
    logging.info("Setting up dataset")
    raw_dataset = setup_dataset(
        dataset_env_name=config.dataset_env_name,
        dataset_env_viewpoint=config.dataset_env_viewpoint,
    )
    raw_dataset = raw_dataset[: config.dataset_n_episodes]

    # Load visual encoder
    logging.info("Setting up visual encoder")
    enc_model_name = config.enc_model_name

    def enc_forward(inputs, is_training=True):
        if enc_model_name.startswith("r3m"):
            resnet_size = int(enc_model_name.split("-")[1])
            model = r3m.R3M(resnet_size)
        else:
            raise ValueError(f"Unknown model name: {enc_model_name}")
        return model(inputs, is_training)

    enc_model = hk.without_apply_rng(hk.transform_with_state(enc_forward))
    enc_state_dict = load_file(
        f"{JAM_MODEL_DIR}/{enc_model_name}/torch_model.safetensors"
    )
    enc_params, enc_state = r3m.load_from_torch_checkpoint(enc_state_dict)

    # Preprocess dataset
    logging.info("Preprocessing dataset")

    @jax.jit
    def preprocess_image(images):
        assert images.dtype == jnp.uint8
        images = (
            images.astype(jnp.float32) - np.asarray(imagenet_util.IMAGENET_MEAN_RGB)
        ) / np.asarray(imagenet_util.IMAGENET_STDDEV_RGB)
        return enc_model.apply(enc_params, enc_state, images, is_training=False)[0]

    def create_dataset(episodes):
        obs = []
        actions = []
        for i, episode in tqdm(enumerate(episodes), desc="Preprocessing images"):
            images = episode["images"]
            embeddings = jax.device_get(preprocess_image(images))
            obs.append(embeddings)
            actions.append(episode["actions"])
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        return tf.data.Dataset.from_tensor_slices((obs, actions))

    # TODO: cache dataset creation
    dataset = create_dataset(raw_dataset)
    # Validate dataset
    _obs, _act = next(iter(dataset))
    assert _obs.shape == (512,)
    assert _act.shape == (4,)

    # Define an MLP policy that operates on R3M embeddings

    logging.info("Setting up policy")

    def _forward_fn(obs):
        return hk.Sequential(
            [
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jnp.tanh,
                hk.nets.MLP(output_sizes=[256, 256, 4], activate_final=False),
            ]
        )(obs)

    model_key, prng_key = jax.random.split(prng_key)
    policy_model = hk.without_apply_rng(hk.transform(_forward_fn))
    policy_params = policy_model.init(model_key, jnp.zeros((1, 512)))

    @jax.jit
    def policy(params, obs):
        return policy_model.apply(params, obs)

    # Define optimizer
    logging.info("Setting up optimizer")
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(policy_params)

    # Define loss function
    logging.info("Setting up loss function")

    def loss_fn(params, batch):
        obs, act = batch
        predicted_actions = policy_model.apply(params, obs)
        return jnp.mean(jnp.square(predicted_actions - act))

    # Define training loop
    logging.info("Setting up training loop")

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Define eval step
    def eval_policy(policy, params, eval_env: Environment, num_episodes: int = 10):
        eval_env = RecordEpisodeStatistics(eval_env)
        eval_env = RecordVideo(
            eval_env,
            video_folder="/tmp/videos",
            # Record 3 videos in the same path
            episode_trigger=lambda x: x == 0,
            video_length=1500,
            name_prefix="eval",
        )
        metrics = {}
        successes = []
        episode_rewards = []
        episode_lengths = []
        elapsed_time = []
        for _ in range(num_episodes):
            _, info = eval_env.reset()
            image = eval_env.render()
            while True:
                action = policy(params, preprocess_image(image[None, ...]))
                _, _, terminated, truncated, info = eval_env.step(np.asarray(action[0]))
                image = eval_env.render()

                if terminated or truncated:
                    successes.append(info["success"])
                    elapsed_time.append(info["episode"]["t"])
                    episode_lengths.append(info["episode"]["l"])
                    episode_rewards.append(info["episode"]["r"])
                    break
        metrics["success_rate"] = np.mean(successes)
        metrics["episode_length"] = np.mean(episode_lengths)
        metrics["episode_reward"] = np.mean(episode_rewards)
        metrics["elapsed_time"] = np.mean(elapsed_time)
        video_path = "/tmp/videos/eval-episode-0.mp4"
        metrics["video"] = wandb.Video(
            video_path, fps=env.metadata["render_fps"], format="mp4"
        )
        return metrics

    # Run training
    logging.info("Running training")
    dataset_iterator = (
        dataset.shuffle(int(1e6), reshuffle_each_iteration=True)
        .repeat()
        .batch(config.batch_size)
        .repeat()
        .as_numpy_iterator()
    )

    # NOTE: Configure WandB through env variables
    # Randomly generate suffix based on time
    import time
    import pathlib

    wandb_dir = pathlib.Path(f"/tmp/{time.time()}")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        group=wandb_config.group,
        name=wandb_config.name,
        dir=str(wandb_dir),
    )

    # Log a few episodes of the dataset
    images = [raw_dataset[i]["images"] for i in range(3)]
    images = np.concatenate(images, axis=0)
    video_path = str(wandb_dir / "dataset.mp4")
    make_video(images, video_path, fps=env.metadata["render_fps"])
    video = wandb.Video(video_path, fps=env.metadata["render_fps"], format="mp4")
    wandb.log({"train/dataset": video})

    for step in tqdm(range(config.train_steps), desc="Training"):
        policy_params, opt_state, loss = train_step(
            policy_params, opt_state, next(dataset_iterator)
        )
        wandb.log({"train/loss": loss})
        if (step + 1) % config.eval_freq == 0:
            eval_metrics = eval_policy(policy, policy_params, env)
            eval_metrics = {"eval/" + k: v for k, v in eval_metrics.items()}
            wandb.log(eval_metrics)

    wandb.finish()
    rm_tree(wandb_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="config")
    parser.add_arguments(WandbConfig, dest="wandb_config")

    args = parser.parse_args()
    run_offline_experiment(args.config, args.wandb_config)
