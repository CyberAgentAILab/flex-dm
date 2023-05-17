# Towards Flexible Multi-modal Document Models (CVPR2023)
This repository is an official implementation of the paper titled above. Please refer to [project page](https://cyberagentailab.github.io/flex-dm/) or [paper](https://arxiv.org/abs/2303.18248) for more details.

## Setup

### Requirements
We check the reproducibility under this environment.
- Python3.7
- CUDA 11.3
- Tensorflow 2.8

### How to install
Install python dependencies. Perhaps this should be done inside `venv`.

```bash
pip install -r requirements.txt
```

Note that Tensorflow has a version-specific system requirement for GPU environment.
Check if the
[compatible CUDA/CuDNN runtime](https://www.tensorflow.org/install/source#gpu) is installed.


## Crello experiments
To try demo on pre-trained models
- download pre-processed datasets for [crello](https://storage.googleapis.com/ailab-public/flexdm/preprocessed_data/crello.zip) / [rico](https://storage.googleapis.com/ailab-public/flexdm/preprocessed_data/rico.zip) and unzip it under `./data`.
- download pre-trained checkpointsfor [crello](https://storage.googleapis.com/ailab-public/flexdm/pretrained_weights/crello.zip) / [rico](https://storage.googleapis.com/ailab-public/flexdm/pretrained_weights/rico.zip) and unzip it under `./results`.

### DEMO
You can test some tasks using the pre-trained models in the [notebook](./notebooks/demo_crello.ipynb).

### Training
You can train your own model.
The trainer script takes a few arguments to control hyperparameters.
See `src/mfp/mfp/args.py` for the list of available options.
If the script slows an out-of-memory error, please make sure other processes do not occupy GPU memory and adjust `--batch_size`.

```bash
bin/train_mfp.sh crello --masking_method random  # Ours-IMP
bin/train_mfp.sh crello --masking_method elem_pos_attr_img_txt  # Ours-EXP
bin/train_mfp.sh crello --masking_method elem_pos_attr_img_txt --weights <WEIGHTS>   # Ours-EXP-FT
```

The trainer outputs logs, evaluation results, and checkpoints to `tmp/mfp/jobs/<job_id>`.
The training progress can be monitored via `tensorboard`.

### Evaluation
You perform quantitative evaluation.
```bash
bin/eval_mfp.sh --job_dir <JOB_DIR> (<ADDITIONAL_ARGS>)
```
See [eval.py](https://github.com/CyberAgentAILab/flex-dm/blob/main/eval.py#L122-L134) for `<ADDITIONAL_ARGS>`.

## RICO experiments

### DEMO
You can test some tasks using the pre-trained models in the [notebook](./notebooks/demo_rico.ipynb).

### Training
The process is almost similar as above.
```bash
bin/train_mfp.sh rico --masking_method random  # Ours-IMP
bin/train_mfp.sh rico --masking_method elem_pos_attr  # Ours-EXP
bin/train_mfp.sh rico --masking_method elem_pos_attr --weights <WEIGHTS>  # Ours-EXP-FT
```

### Evaluation
The process is similar as above.

## Citation

If you find this code useful for your research, please cite our paper.

```
@inproceedings{inoue2023document,
    title={{Towards Flexible Multi-modal Document Models}},
    author={Naoto Inoue and Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023},
    pages={14287-14296},
  }
```
