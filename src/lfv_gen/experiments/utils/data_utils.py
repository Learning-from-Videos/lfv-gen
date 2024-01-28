from lfv_gen.envs.setup_metaworld_env import (
    MetaworldEnvName,
    MetaworldCameraName,
)


def get_dataset_name(
    suite: str,
    env_name: MetaworldEnvName,
    camera_name: MetaworldCameraName,
):
    return f"{suite}-{env_name}-{camera_name}"
