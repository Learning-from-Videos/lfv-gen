from lfv_gen.core.types import GymEnv

import gymnasium
import numpy as np
from gymnasium.wrappers import PixelObservationWrapper


class PixelObservationDictToBoxWrapper(gymnasium.ObservationWrapper):
    """Wraps a PixelObservationWrapper so that the observation is a Box instead of a Dict"""

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        pixel_obs_space = self.observation_space.spaces["pixels"]
        self.observation_space = pixel_obs_space

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return observation["pixels"]


def wrap_env_as_pixel(
    env: GymEnv,
    pixels_only: bool = False,
) -> GymEnv:
    env = PixelObservationWrapper(env, pixels_only=pixels_only)
    return env
