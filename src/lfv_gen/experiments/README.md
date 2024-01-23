# Learning from videos experiments


## Setup

1. Ensure you have downloaded the datasets and checkpoints.
2. Ensure environment variables are configured appropriately:
```bash
export JAM_MODEL_DIR=/path/to/jam/data/models
export MUJOCO_GL=egl
export WANDB_PROJECT=some_project
export WANDB_ENTITY=some_entity
```

## Running experiments

To run a single experiment: 
```bash
python -m lfv_gen.experiments.offline_experiment
```
To run sweeps:
```bash
python -m lfv_gen.experiments.run_sweep
```

For help, add `--help` to both commands above

## Proof-of-concept (deprecated)

The `r3m_minimal` folder is a self-contained proof of concept of R3M + BC. 