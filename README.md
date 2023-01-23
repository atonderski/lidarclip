# LidarCLIP

This is the official implementation of [LidarCLIP or: How I Learned to Talk to Point Clouds](https://arxiv.org/abs/2212.06858).

## Checkpoints

| CLIP Version | Training Dataset | Link                                                                              |
| ------------ | ---------------- | --------------------------------------------------------------------------------- |
| ViT-L/14     | ONCE             | [google drive](https://drive.google.com/file/d/1ANImRlmcZ3Yoa1jbLrQO57M0-T2iIbL2) |
| ViT-B/32     | ONCE             | [google drive](https://drive.google.com/file/d/1eNsfQDz7TYT2UCYhxKQcWIZMfAuAHkUA) |


## Instructions

- download the [SST](https://github.com/tusen-ai/SST) submodule `git submodule update --recursive`
- build the dockerfile `docker build -t lidarclip -f docker/Dockerfile .`
- spin up a container with access to the dataset and at least one gpu. (`docker run <...> lidarclip`)
- in the docker container, run `python train.py --datadir=<dataset-path> --checkpoint-save-dir=<checkpoint-dir> --name=<experiment-name>`. You can specify many additional flags, here is an example command: `--name lidarclip-main --batch-size 128 --workers 4 --checkpoint-save-dir /proj/lidarclip/checkpoints/ --clip-model ViT-L/14`.
- pre-compute LidarCLIP features: `python.py scripts/cache_embeddings.py --checkpoint=<checkpoint-dir>/<checkpoint-file>`. By default these are for the once validation set, but it is easily changed with the `--dataset-name`, `--data-path`, and `--split` arguments.
- now you can explore the capabilities of the model by running one of the notebooks, placed under `notebooks/`:
    - `retrieval_and_zero_shot.ipynb` allows for qualitative exploration of retrieval and zero-shot classification
    - `retrieval_metric.ipynb` is for computing quantitative retrieval metrics
    - `diffusion/CLIP_Guided_Stable_Diffusion.ipynb` is for lidar-to-image generation
    - `generate_captions.ipynb` is for lidar captioning. Note that you need to train LidarCLIP against ViT-B/32 (instead of the default which is ViT-L/14).

Note that some paths have to be modified in the notebooks to point your desired dataset and cached features.

## Dataset preparation

We refer to the official download and preparation instructions for [ONCE](https://once-for-auto-driving.github.io/download.html) and [NuScenes](https://nuscenes.org/nuscenes#download). Once the dataset directories are set up according to the official instructions, our code works without any additional steps. Note that NuScenes is entirely optional and only used for evaluating domain shift capabilities.
