# LfV-Gen

A repository to evaluate the generalization properties of learning-from-videos methods

TODOs:
- [X] Need to add custom cameras to Metaworld HEAD
- [X] Need to add instructions / script to download the dataset
- [ ] Need to add instructions / script to download the pre-trained R3M checkpoints

Experiments to run:
1. Test task generalization
2. Test different training, testing viewpoints

## Robot Datasets
We provide a list of robot demonstration datasets in `lfv_gen.data.dataset_zoo`

To download all datasets, run:
```bash
python -m lfv_gen.scripts.download_data
```

At the moment, only R3M datasets are supported 

## Pre-trained Models
To evaluate pre-trained model checkpoints, we rely on [jam](https://github.com/ethanluoyc/jam)

Follow the instructions there to download datasets. 
