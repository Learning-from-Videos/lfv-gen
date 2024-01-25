import gymnasium
import wandb
import numpy as np
import jax.numpy as jnp

from dataclasses import dataclass
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from typing import Literal, Callable
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

# common types
Environment = gymnasium.Env

env_viewpoint = Literal[
    "top_cap2",
    "left_cap2",
    "right_cap2",
]


class Policy:
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Input: observation (no batch dimension)
        Output: action (no batch dimension)
        """
        raise NotImplementedError


@dataclass
class JaxPolicy:
    policy_fn: Callable[[jnp.ndarray], jnp.ndarray]
    params: jnp.ndarray

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        act_jnp = self.policy_fn(self.params, obs[None, ...])[0]
        return np.asarray(act_jnp)


def setup_env(env_name: str, env_viewpoint: str) -> Environment:
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    e = env_cls(render_mode="rgb_array")

    # Hack: enable random reset
    e._freeze_rand_vec = False

    # Hack: set the camera view  to be the same as R3M
    e.camera_name = env_viewpoint
    # Hack: set the renderer to return 256x256 images
    e.model.vis.global_.offwidth = 256
    e.model.vis.global_.offheight = 256

    # Simple validation
    e.reset()
    image_array = e.render()
    assert image_array.shape == (256, 256, 3)
    return e


def eval_policy(policy: Policy, eval_env: Environment, num_episodes: int = 10):
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
        obs, info = eval_env.reset()
        while True:
            action = policy(obs)
            obs, _, terminated, truncated, info = eval_env.step(action)
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
        video_path, fps=eval_env.metadata["render_fps"], format="mp4"
    )
    return metrics
