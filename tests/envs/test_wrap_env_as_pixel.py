import numpy as np

from lfv_gen.envs.setup_metaworld_env import setup_metaworld_env
from lfv_gen.envs.wrap_env_as_pixel import wrap_env_as_pixel


def test_wrap_env_as_pixel():
    env = setup_metaworld_env(
        env_name="drawer-open-v2-goal-observable",
        camera_name="top_cap2",
        seed=0,
    )
    env = wrap_env_as_pixel(env)
    obs, _ = env.reset()
    image = env.render()

    assert obs.shape == image.shape
    assert obs.dtype == image.dtype
    assert np.allclose(obs, image)
