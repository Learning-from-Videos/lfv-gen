import pathlib
import wandb
import uuid
import shutil

from dataclasses import dataclass
from typing import Any


@dataclass
class WandbConfig:
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    name: str | None = None


class WandbLogger:
    def __init__(
        self, wandb_config: WandbConfig = WandbConfig(), delete_logs_after_finish=True
    ):
        self.wandb_dir = pathlib.Path(f"/tmp/{uuid.uuid4()}")
        self.wandb_dir.mkdir(parents=True, exist_ok=True)
        self.delete_logs_after_finish = delete_logs_after_finish
        wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            group=wandb_config.group,
            name=wandb_config.name,
            dir=str(self.wandb_dir),
        )

    def log(self, metrics: dict[str, Any]):
        wandb.log(metrics)

    def __del__(self):
        wandb.finish()
        if self.delete_logs_after_finish:
            # Recursively delete wandb dir
            shutil.rmtree(self.wandb_dir)
