import numpy as np
from lfv_gen.envs.setup_metaworld_env import setup_metaworld_env


def test_seeded_env_creation_is_deterministic():
    env1 = setup_metaworld_env("drawer-open-v2-goal-observable", "top_cap2", seed=0)
    obs1, _ = env1.reset()
    img1 = env1.render()

    env2 = setup_metaworld_env("drawer-open-v2-goal-observable", "top_cap2", seed=0)
    obs2, _ = env2.reset()
    img2 = env2.render()

    assert np.allclose(obs1, obs2)
    assert np.allclose(img1, img2)


def test_env_image_is_256x256():
    env = setup_metaworld_env("drawer-open-v2-goal-observable", "top_cap2", seed=0)
    env.reset()
    img = env.render()
    assert img.shape == (256, 256, 3)
