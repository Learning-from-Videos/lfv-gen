import metaworld.policies as POLICIES

from collections import namedtuple
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.policies.policy import Policy as SawyerXYZPolicy

from typing import Literal
from lfv_gen.core.types import GymEnv
from lfv_gen.core.policy import Policy

MetaworldEnvName = str
MetaworldCameraName = Literal[
    # camera name defined in:
    # metaworld/envs/assets_v2/objects/assets/xyz_base.xml
    "corner",
    "corner2",
    "corner3",
    "top_cap2",
    "left_cap2",
    "right_cap2",
]


def setup_metaworld_env(
    env_name: MetaworldEnvName,
    camera_name: MetaworldCameraName,
    seed: int = 0,
) -> GymEnv:
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    e: SawyerXYZEnv = env_cls(render_mode="rgb_array")

    # Hack: enable random reset
    e._freeze_rand_vec = False
    e.seed(seed)

    # Hack: add a DmEnv-like spec
    e.spec = namedtuple("spec", ["id", "max_episode_steps"])
    e.spec.id = env_name
    e.spec.max_episode_steps = 500

    # Hack: set the camera view  to be the same as R3M
    e.camera_name = camera_name
    # Hack: set the renderer to return 256x256 images
    e.model.vis.global_.offwidth = 256
    e.model.vis.global_.offheight = 256

    return e


def _get_env_base_name(env_name: MetaworldEnvName) -> str:
    """Get base name of environment

    For example, "drawer-open-v2-goal-observable" -> "drawer-open-v2"
    """
    return env_name.replace("-goal-observable", "").replace("-goal-hidden", "")


def setup_metaworld_policy(env_name: MetaworldEnvName) -> Policy:
    """Get scripted expert policy"""
    env_base_name = _get_env_base_name(env_name)
    env_cls = ALL_V2_ENVIRONMENTS[env_base_name]
    env_cls_name = env_cls.__name__
    policy_cls_name = env_cls_name.replace("Env", "") + "Policy"
    policy_cls: SawyerXYZPolicy = getattr(POLICIES, policy_cls_name)
    policy = policy_cls()
    return lambda obs: policy.get_action(obs).astype(obs.dtype)
