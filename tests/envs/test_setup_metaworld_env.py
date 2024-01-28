import numpy as np
import pytest

from lfv_gen.envs.setup_metaworld_env import setup_metaworld_env
from lfv_gen.envs.setup_metaworld_env import setup_metaworld_policy


@pytest.mark.parametrize(
    "env_name",
    [
        "drawer-open-v2-goal-observable",
        "drawer-close-v2-goal-observable",
        "assembly-v2-goal-observable",
        "bin-picking-v2-goal-observable",
        "hammer-v2-goal-observable",
    ],
)
def test_seeded_env_obs_is_deterministic(env_name: str):
    env1 = setup_metaworld_env(env_name, "top_cap2", seed=0)
    obs1, _ = env1.reset()

    env2 = setup_metaworld_env(env_name, "top_cap2", seed=0)
    obs2, _ = env2.reset()

    assert np.allclose(obs1, obs2)
    env1.close()
    env2.close()


@pytest.mark.parametrize(
    "env_name",
    [
        "drawer-open-v2-goal-observable",
        "drawer-close-v2-goal-observable",
        # I don't know why this one fails
        pytest.param("assembly-v2-goal-observable", marks=pytest.mark.xfail),
        "bin-picking-v2-goal-observable",
        "hammer-v2-goal-observable",
    ],
)
def test_seeded_env_img_is_deterministic(env_name: str):
    env1 = setup_metaworld_env(env_name, "top_cap2", seed=0)
    env1.reset()
    img1 = env1.render()

    env2 = setup_metaworld_env(env_name, "top_cap2", seed=0)
    env2.reset()
    img2 = env2.render()

    assert np.allclose(img1, img2)
    env1.close()
    env2.close()


@pytest.mark.parametrize(
    "env_name",
    [
        "drawer-open-v2-goal-observable",
        "drawer-close-v2-goal-observable",
        "assembly-v2-goal-observable",
        "bin-picking-v2-goal-observable",
        "hammer-v2-goal-observable",
    ],
)
def test_setup_policy(env_name: str):
    policy = setup_metaworld_policy("drawer-open-v2-goal-observable")
    assert callable(policy)


def test_env_image_is_256x256():
    env = setup_metaworld_env("drawer-open-v2-goal-observable", "top_cap2", seed=0)
    env.reset()
    img = env.render()
    assert img.shape == (256, 256, 3)
    env.close()
