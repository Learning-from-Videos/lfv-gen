import numpy as np

from dm_env_wrappers import GymnasiumWrapper
from lfv_gen.envs.setup_metaworld_env import setup_metaworld_env


def test_dm_env_wrapper_spec():
    gym_env = setup_metaworld_env("drawer-open-v2-goal-observable", "top_cap2", seed=0)
    env = GymnasiumWrapper(gym_env)
    assert env.observation_spec().dtype == gym_env.observation_space.dtype
    assert env.observation_spec().shape == gym_env.observation_space.shape
    assert env.action_spec().dtype == gym_env.action_space.dtype
    assert env.action_spec().shape == gym_env.action_space.shape
    assert env.reward_spec().dtype == np.float64
    assert env.reward_spec().shape == ()
    timestep = env.reset()
    assert timestep.observation.dtype == gym_env.observation_space.dtype
    env.close()
