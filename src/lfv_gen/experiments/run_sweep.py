import simple_parsing
from dataclasses import dataclass, replace
from lfv_gen.experiments.offline_experiment import (
    run_offline_experiment, 
    ExperimentConfig, 
    WandbConfig
)
from typing import Iterable, Any

def get_dataset_size_sweep(config: ExperimentConfig) -> Iterable[ExperimentConfig]:
    for dataset_size in [20, 50, 100, 200]:
        yield replace(config, dataset_size=dataset_size)

def get_enc_model_sweep(config: ExperimentConfig) -> Iterable[ExperimentConfig]:
    for enc_model_name in [
        "r3m/r3m-18",
        "r3m/r3m-34",
        "r3m/r3m-50",
    ]:
        yield replace(config, enc_model_name=enc_model_name)

def get_eval_env_and_dataset_env_viewpoint_sweep(config: ExperimentConfig) -> Iterable[ExperimentConfig]:
    for viewpoint in ["top_cap2", "right_cap2", "left_cap2"]:
        yield replace(
            config, 
            dataset_env_viewpoint=viewpoint, 
            eval_env_viewpoint=viewpoint
        )

def get_eval_env_viewpoint_sweep(config: ExperimentConfig) -> Iterable[ExperimentConfig]:
    for viewpoint in ["top_cap2", "right_cap2", "left_cap2"]:
        yield replace(
            config, 
            eval_env_viewpoint=viewpoint
        )

def get_eval_and_dataset_env_name_sweep(config: ExperimentConfig) -> Iterable[ExperimentConfig]:
    """ Sweep over the environments used for training and evaluation. 
    
    In all cases we keep the dataset_env_name the same as the eval_env_name.
    """
    for env_name in [
        "assembly-v2-goal-observable", 
        "bin-picking-v2-goal-observable",
        "button-press-topdown-v2-goal-observable",
        "drawer-open-v2-goal-observable",
        "hammer-v2-goal-observable",
    ]:
        yield replace(config, dataset_env_name=env_name, eval_env_name=env_name)

def get_eval_env_name_sweep(config: ExperimentConfig) -> Iterable[ExperimentConfig]:
    """ Sweep over the environments used for evaluation.
    
    Here we keep the dataset_env_name constant to evaluate cross-task transfer. """
    for env_name in [
        "assembly-v2-goal-observable", 
        "bin-picking-v2-goal-observable",
        "button-press-topdown-v2-goal-observable",
        "drawer-open-v2-goal-observable",
        "hammer-v2-goal-observable",
    ]:
        yield replace(config, eval_env_name=env_name)

# Dictionary of sweep_name: (sweep_fn, sweep_var_name)
sweeps: dict[str, tuple[Any, str]] = {
    "dataset_size": (get_dataset_size_sweep, "dataset_size"),
    "enc_model": (get_enc_model_sweep, "enc_model_name"),
    "eval_env_and_dataset_env_viewpoint": (get_eval_env_and_dataset_env_viewpoint_sweep, "dataset_env_viewpoint"),
    "eval_env_viewpoint": (get_eval_env_viewpoint_sweep, "eval_env_viewpoint"),
    "eval_and_dataset_env_name": (get_eval_and_dataset_env_name_sweep, "dataset_env_name"),
    "eval_env_name": (get_eval_env_name_sweep, "eval_env_name"),
}

if __name__ == "__main__":

    parser = simple_parsing.ArgumentParser()
    parser.add_argument(
        "--sweep-name", 
        type=str, 
        default="eval_env_and_dataset_env_viewpoint",
        choices = list(sweeps.keys()),
    )
    parser.add_arguments(ExperimentConfig, dest="config")
    args = parser.parse_args()

    # Run sweep
    sweep_fn, sweep_var = sweeps[args.sweep_name]
    for config in sweep_fn(args.config):
        wandb_config = WandbConfig(
            group=args.sweep_name,
            name=f"{sweep_var}={getattr(config, sweep_var)}",
        )
        run_offline_experiment(config, wandb_config=wandb_config)