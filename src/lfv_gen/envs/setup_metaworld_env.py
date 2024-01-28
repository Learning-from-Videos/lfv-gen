from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv

from typing import Literal
from lfv_gen.core.types import GymEnv

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

    # Hack: set the camera view  to be the same as R3M
    e.camera_name = camera_name
    # Hack: set the renderer to return 256x256 images
    e.model.vis.global_.offwidth = 256
    e.model.vis.global_.offheight = 256

    return e
