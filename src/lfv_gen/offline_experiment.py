from collections import namedtuple
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from lfv_gen.config import ROOT_DIR

import wandb
import gymnasium
import tensorflow as tf
import pickle
import numpy as np
from typing import Any
from moviepy.editor import ImageSequenceClip

import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
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
JAM_MODEL_DIR = os.getenv("JAM_MODEL_DIR")

# common types
Environment = gymnasium.Env
Dataset = tf.data.Dataset

@dataclass
class ExperimentConfig:
    dataset_env_name: str
    dataset_viewpoint: str
    eval_env_name: str
    eval_viewpoint: str
    enc_model_name: str
    train_steps: int
    eval_freq: int
    batch_size: int
    learning_rate: float
    seed: int

def make_gif(image_array: np.ndarray, gif_path: str, fps: int):
    """
    Convert a numpy array of images to a GIF.

    Parameters:
    image_array (numpy.ndarray): An array of images of shape (batch, height, width, channels).
    gif_path (str): Path to save the GIF.
    fps (int): Frames per second in the GIF.
    """
    images = [img for img in image_array]
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_gif(gif_path)

def run_offline_experiment():

    def setup_eval_env(
        eval_env_name: str,
        eval_env_viewpoint: str
    ) -> Environment:
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
        e  = env_cls(render_mode="rgb_array")

        # Hack: enable random reset
        e._freeze_rand_vec = False

        # Hack: create an env spec similar to dm.Env
        e.spec = namedtuple('spec', ['id', 'max_episode_steps'])
        e.spec.id = env_name
        e.spec.max_episode_steps = 500

        # Hack: set the camera view  to be the same as R3M
        e.camera_name = "top_cap2"
        # Hack: set the renderer to return 256x256 images
        e.model.vis.global_.offwidth = 256
        e.model.vis.global_.offheight = 256

        # Simple validation
        e.reset()
        image_array = e.render()
        assert image_array.shape == (256, 256, 3)
        return e
    
    def setup_dataset(dataset_path: str) -> Dataset:
        with open(ROOT_DIR / dataset_path, "rb") as f:
            dataset: list[dict[str, Any]] = pickle.load(f)

        # Simple validation 
        assert 'images' in dataset[0].keys()
        assert 'actions' in dataset[0].keys()
        assert 'observations' in dataset[0].keys()
        assert 'rewards' in dataset[0].keys()
        assert dataset[0]['images'].shape == (500, 256, 256, 3)
        return dataset
    
    # Load environment
    logging.info("Setting up environment")
    env_name = "drawer-open-v2-goal-observable"
    env: Environment = setup_env(env_name)

    # Load dataset
    logging.info("Setting up dataset")
    dataset_path = f"datasets/metaworld/top_cap2/{env_name}.pickle"
    raw_dataset = setup_dataset(dataset_path)
    
    # Load visual encoder
    logging.info("Setting up visual encoder")    
    def r3m_forward(inputs, resnet_size: int = 18, is_training=True):
        model = r3m.R3M(resnet_size)
        return model(inputs, is_training)
    
    enc_model_name = "r3m/r3m-18"
    enc_model = hk.without_apply_rng(hk.transform_with_state(r3m_forward))
    # TODO: un-hardcode path
    enc_state_dict = load_file(f"{JAM_MODEL_DIR}/{enc_model_name}/torch_model.safetensors")
    enc_params, enc_state = r3m.load_from_torch_checkpoint(enc_state_dict)

    # Preprocess dataset
    logging.info("Preprocessing dataset")
    @jax.jit
    def preprocess_image(images):
        assert images.dtype == jnp.uint8
        images = (images.astype(jnp.float32) - np.asarray(imagenet_util.IMAGENET_MEAN_RGB)) / np.asarray(imagenet_util.IMAGENET_STDDEV_RGB)
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
    policy_model = hk.without_apply_rng(hk.transform(_forward_fn))
    policy_params = policy_model.init(jax.random.PRNGKey(42), jnp.zeros((1, 512)))
    @jax.jit
    def policy(params, obs):
        return policy_model.apply(params, obs)

    # Define optimizer
    logging.info("Setting up optimizer")
    optimizer = optax.adam(3e-4)
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
            video_folder=f"videos",
            # Record 3 videos in the same path
            episode_trigger= lambda x: x == 0,
            video_length=1500,
            name_prefix=f"{env_name}"
        )
        metrics = {}
        successes = []
        episode_rewards = []
        episode_lengths = []
        elapsed_time  = []
        for _ in range(num_episodes):
            _, info = eval_env.reset()
            image = eval_env.render()
            while True:
                action = policy(params, preprocess_image(image[None, ...]))
                _, _, terminated, truncated, info = eval_env.step(np.asarray(action[0]))
                image = eval_env.render()
                
                if terminated or truncated:
                    successes.append(info['success'])
                    elapsed_time.append(info['episode']['t'])
                    episode_lengths.append(info['episode']['l'])
                    episode_rewards.append(info['episode']['r'])
                    break
        metrics['success_rate'] = np.mean(successes)
        metrics['episode_length'] = np.mean(episode_lengths)
        metrics['episode_reward'] = np.mean(episode_rewards)
        metrics['elapsed_time'] = np.mean(elapsed_time)
        return metrics
    
    # Run training
    logging.info("Running training")
    dataset_iterator = (
        dataset
        .shuffle(int(1e6), reshuffle_each_iteration=True)
        .repeat()
        .batch(256)
        .repeat()
        .as_numpy_iterator()
    )
    
    # NOTE: Configure WandB through env variables
    # Randomly generate suffix based on time
    import time 
    wandb.init(dir=f"/tmp/wandb/{time.time()}")

    num_train_steps = 10000
    eval_freq = 1000

    for step in range(num_train_steps):
        policy_params, opt_state, loss = train_step(policy_params, opt_state, next(dataset_iterator))
        wandb.log({"train/loss": loss})
        if step % eval_freq == 0:
            eval_metrics = eval_policy(policy, policy_params, env)
            eval_metrics={"eval/" + k: v for k, v in eval_metrics.items()}
            wandb.log(eval_metrics)

    wandb.finish()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_offline_experiment()