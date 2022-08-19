# Wild-Time: A Benchmark of in-the-Wild Distribution Shifts over Time

**Note**: This is a preliminary version of the Wild-Time benchmark. We are working on code refactoring and will release the final version in 1-2 months.

## Overview
Distribution shifts occur when the test distribution differs from the training distribution, and can considerably degrade performance of machine learning models deployed in the real world. While recent works have studied robustness to distribution shifts, distribution shifts arising from the passage of time have the additional structure of timestamp metadata. Real-world examples of such shifts are underexplored, and it is unclear whether existing models can leverage trends in past distribution shifts to reliably extrapolate into the future. To address this gap, we curate Wild-Time, a benchmark of 7 datasets that reflect temporal distribution shifts arising in a variety of real-world applications, including drug discovery, patient prognosis, and news classification.

![Wild-Time -- Dataset Description](data_description.png)

This repo includes scripts to download all Wild-Time datasets, code for all baselines, and scripts for training and evaluating these baselines on Wild-Time datasets.

If you find this repository useful in your research, please cite the following paper:

```
@inproceedings{yao2022wildtime,
  title={Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time},
  author={Huaxiu Yao and Caroline Choi and Yoonho Lee and Pang Wei Koh and Chelsea Finn},
  booktitle={ICML 2022 Shift Happens Workshop},
  year={2022},
}
```

We will release the arXiv version of our paper, along with the final code repository, in 1-2 months.

## Installation
To download the Wild-Time datasets and run baselines, please clone this repo to your local machine.

```
git clone git@github.com:huaxiuyao/Wild-Time.git
cd Wild-Time
```
In the directory `Wild-Time`, create the folders `Data`, `checkpoints`, and `results`.

### Requirements

- numpy=1.19.1
- pytorch=1.11.0
- pytorch-tabula=0.7.0
- pytorch-transformers=1.2.0
- pytorch-lightning=1.5.9
- pandas=1.4.2
- huggingface-hub=0.5.1
- PyTDC=0.3.6

### Downloading the Wild-Time datasets

Create the folder `Wild-Time/Data`.

To download the ArXiv, Drug-BA, FMoW, HuffPost, Precipitation, and Yearbook datasets, run the command `python download_datasets.py`. To download the MIMIC dataset, please read the next section.

### Accessing the MIMIC-IV dataset

Due to patient confidentiality, users must be credentialed on PhysioNet and sign the Data Use Agreement before accessing the MIMIC-IV dataset.
Here are instructions for how to do so.

1. Become a credentialed user on PhysioNet. This means that you must formally submit your personal details for review, so that PhysioNet can confirm your identity.
  - If you do not have a PhysioNet account, register for one [here](https://physionet.org/register/).
  - Follow these [instructions](https://physionet.org/settings/credentialing/}{instructions) for credentialing on PhysioNet.
  - Complete the "Data or Specimens Only Research" [training course](https://physionet.org/about/citi-course/).
2. Sign the data use agreement.
  - [Log in](https://physionet.org/login/) to your PhysioNet account.
  - Go to the MIMIC-IV dataset [project page](https://physionet.org/content/mimiciv/2.0/).
  - Locate the "Files" section in the project description.
  - Click through, read, and sign the Data Use Agreement (DUA).
4. Go to https://physionet.org/content/mimiciv/1.0/ and download the following CSV files from the "core" and "hosp" modules to `./Data`:
    - patients.csv
    - admissions.csv
    - diagnoses_icd.csv
    - procedures_icd.csv
   Decompress the files and put them under `./Data/raw/mimic4`.
4. Run the command `python ../src/data/mimic/preprocess.py` to get the Wild-Time MIMIC datasets: `mimic_preprocessed_readmission.pkl` and `mimic_preprocessed_mortality.pkl`.

## Running baselines

To train a baseline on a Wild-Time dataset and evaluate under Eval-Fix (default evaluation), use the command
```
python main.py --dataset=[DATASET] --method=[BASELINE] --lr=[LEARNING RATE] --train_update_iters=[TRAIN ITERS] --num_workers=[WORKERS] --random_seed=[SEED] --offline --split_time=[TIME STEP] [BASELINE-SPECIFIC HYPERPARAMETERS]
```

- Specify the dataset with `--dataset`.
  - [arxiv, drug, fmow, huffpost, mimic, precipitation, yearbook]
  - For MIMIC, specify one of two prediction tasks (mortality and readmission) using`--prediction_type=mortality` or `--prediction_type=readmission`.
- Specify the baseline with `--method`.
- To run Eval-Fix, add the flag `--offline`.
  - Specify the ID/OOD split time step with `--split_time`.
- To run Eval-Stream, add the flag `--eval_next_timesteps`.
- Set the training batch size with `--mini_batch_size`.
- Set the number of training iterations with `--train_update_iters`.
- [Optional] If using a data directory or checkpoint directory other than `./Data` and `./checkpoints`, specify their paths with `--data_dir` and `--log_dir`. 

### CORAL
- Set the number of groups (e.g., number of time windows) with `--num_groups`.
- Set the group size (e.g., length of each time window) with `--group_size`.
- Specify the weight of the CORAL loss with `--coral_lambda` (default: 1.0).
- Add `--non_overlapping` to sample from non-overlapping time windows.

Example command:
```
python main.py --dataset=arxiv --method=coral --offline --split_time=2016 --coral_lambda=0.9 --num_groups=4 --group_size=4 --mini_batch_size=64 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --num_workers=8 --random_seed=1 --log_dir=./checkpoints
```

#### GroupDRO
- Set the number of groups (e.g., number of time windows) with `--num_groups`.
- Set the group size (e.g., length of each time window) with `--group_size`.
- Add `--non_overlapping` to sample from non-overlapping time windows.

Example command:
```
python main.py --dataset=drug --method=groupdro --offline --split_time=2016 --num_groups=3 --group_size=2 --mini_batch_size=256 --train_update_iter=5000 --lr=2e-5 --random_seed=1 --log_dir=./checkpoints --data_dir=./Data/Drug-BA
```

### IRM
- Set the number of groups (e.g., number of time windows) with `--num_groups`.
- Set the group size (e.g., length of each time window) with `--group_size`.
- Specify the weight of the IRM penalty loss with `--irm_lambda` (default: 1.0)
- Specify the number of iterations after which to anneal the IRM penalty loss wtih `--irm_penalty_anneal_iters` (default: 0).

Example command:
```
python main.py --dataset=fmow --method=irm --offline --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=3 --group_size=3 --mini_batch_size=64 --train_update_iter=3000 --lr=1e-4 --weight_decay=0.0 --split_time=10 --num_workers=8 --random_seed=1 --log_dir=./checkpoints
```

### ERM

Example command:
```
python main.py --dataset=huffpost --method=erm --offline --mini_batch_size=32 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --split_time=2015 --num_workers=8 --random_seed=1 --log_dir=./checkpoints
```

### LISA
- Set the interpolation ratio $\lambda \sim Beta(\alpha, \alpha)$ by specifying $\alpha$ with `--mix_alpha` (default: 2.0).

Example command:
```
python main.py --dataset=mimic --method=erm --offline --lisa --mix_alpha=2.0 --prediction_type=mortality --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --random_seed=1 --split_time=2013 --data_dir=./Data --log_dir=./checkpoints/
```

### Mixup
- Set the interpolation ratio $\lambda \sim Beta(\alpha, \alpha)$ by specifying $\alpha$ with `--mix_alpha` (default: 2.0).

Example command:
```
python main.py --dataset=mimic --method=erm --offline --mixup --mix_alpha=2.0 --prediction_type=readmission --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --random_seed=1 --split_time=2013 --data_dir=./Data --log_dir=./checkpoints/
```

### Averaged Gradient Episodic Memory (A-GEM)
- Set the buffer size with `--buffer_size` (default: 1000).

Example command:
```
python main.py --dataset=precipitation --method=agem --buffer_size=1000 --offline --mini_batch_size=128 --train_update_iter=5000 --lr=0.001 --split_time=7 --random_seed=1 --log_dir=./checkpoints
```

### (Online) Elastic Weight Consolidation (EWC)
- Set the regularization strength (e.g., weight of the EWC loss) with `ewc_lambda` (default: 1.0). 

Sample command:
```
python main.py --dataset=yearbook --method=ewc --ewc_lambda=0.5 --online --mini_batch_size=32 --train_update_iter=3000 --lr=0.001 --offline --split_time=1970 --random_seed=1 --log_dir=./checkpoints
```

### Fine-tuning (FT)
Example command:
```
python main.py --dataset=arxiv --method=ft --mini_batch_size=64 --train_update_iter=1000 --lr=2e-5 --weight_decay=0.01 --offline --split_time=2016 --num_workers=8 --random_seed=1 --log_dir=./checkpoints
```

### Synaptic Intelligence (SI)
- Set the SI regularization strength with `--si_c` (default: 0.1).
- Set the dampening parameter with `--epsilon` (default: 0.001).

Example command:
```
python main.py --dataset=drug --method=si --si_c=0.1 --epsilon=0.001 --lr=5e-5 --mini_batch_size=256 --train_update_iter=5000 --split_time=2016 --random_seed=1 --log_dir=./checkpoints --data_dir=./Data/Drug-BA
```

### SimCLR
- Specify the number of iterations for which to learn representations using SimCLR with `--train_update_iter`.
- Specify the number of iterations to finetune the classifier with `--finetune_iter`.

Example command:
```
python main.py --dataset=fmow --method=simclr --offline --mini_batch_size=64 --train_update_iter=1500 --finetune_iter=1500 --lr=1e-4 --weight_decay=0.0 --split_time=13 --num_workers=8 --random_seed=1
```

### SwaV
- Specify the number of iterations for which to learn representations using SimCLR with `--train_update_iter`.
- Specify the number of iterations to finetune the classifier with `--finetune_iter`.

Example command:
```
python main.py --dataset=yearbook --method=swav --mini_batch_size=32 --train_update_iter=2700 --finetune_iter=300 --lr=0.001 --offline --random_seed=1 --split_time=1970
```

### Stochastic Weighted Averaging (SWA)

Example command:
```
python main.py --dataset=huffpost --method=swa --offline --mini_batch_size=32 --train_update_iter=6000 --lr=2e-5 --weight_decay=0.01 --split_time=2015 --num_workers=8 --random_seed=1 --log_dir=./checkpoints
```

## Scripts
In `scripts/`, we provide a set of scripts that can be used to train and evaluate models on the Wild-Time datasets. These scripts contain the hyperparameter settings used to benchmark the baselines in our paper.

All Eval-Fix scripts can be found located under `scripts/eval-fix`. All Eval-Stream scripts are located under under `scripts/eval-stream`.

## Checkpoints
In `checkpoints/`, we provide pretrained model checkpoints for all baselines used in our paper under the Eval-Fix train setting.

## Licenses
We list the licenses for each Wild-Time dataset below:

- Yearbook: MIT License
- FMoW: [The Functional Map of the World Challenge Public License](https://raw.githubusercontent.com/fMoW/dataset/master/LICENSE)
- MIMIC-IV (Readmission and Mortality): [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimiciv/view-license/0.4/)
- Drug-BA: MIT License
- Precipitation: CC BY-NC 4.0
- Huffpost: CC0: Public Domain
- arXiv: CC0: Public Domain


## Acknowledgements

We thank the authors of all baselines. Most of our implementations follow the corresponding original released versions. We thank Zhenbang Wu for his assistance in preprocessing the MIMIC-IV dataset.
