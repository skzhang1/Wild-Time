# Wild-Time: A Benchmark of in-the-Wild Distribution Shifts over Time

**Note**: This is a preliminary version of the Wild-Time benchmark. We are working on code refactoring and will release the final version in 1-2 months.

## Overview
Distribution shifts occur when the test distribution differs from the training distribution, and can considerably degrade performance of machine learning models deployed in the real world. While recent works have studied robustness to distribution shifts, distribution shifts arising from the passage of time have the additional structure of timestamp metadata. Real-world examples of such shifts are underexplored, and it is unclear whether existing models can leverage trends in past distribution shifts to reliably extrapolate into the future. To address this gap, we curate Wild-Time, a benchmark of 7 datasets that reflect temporal distribution shifts arising in a variety of real-world applications, including drug discovery, patient prognosis, and news classification.

![Wild-Time -- Dataset Description](data_description.png)

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

## Prerequisites

- numpy=1.19.1
- pytorch=1.11.0
- pytorch-tabula=0.7.0
- pytorch-transformers=1.2.0
- pytorch-lightning=1.5.9
- pandas=1.4.2
- huggingface-hub=0.5.1
- PyTDC=0.3.6

## Downloading the Wild-Time Datasets

First, create the folder `./Data`.

To download the `ArXiv`, `Drug-BA`, `FMoW`, `HuffPost`, `Precipitation`, and `Yearbook` datasets, run the command `python download_datasets.py`.

### Accessing the MIMIC-IV Dataset

Due to patient confidentiality, users must be credentialed on PhysioNet and sign the DUA before downloading the MIMIC-IV dataset.
Here are instructions for how to do so.

1. Obtain 
  - INSTRUCTIONS [TO DO: ADD THIS]
2. 
3. Go to https://physionet.org/content/mimiciv/1.0/ and download the following CSV files from the "core" and "hosp" modules to `./Data:
    - patients.csv
    - admissions.csv
    - diagnoses_icd.csv
    - procedures_icd.csv
   Decompress the files and put them under `./Data/raw/mimic4`
4. Run the command `python ../src/data/mimic/preprocess.py` to get `mimic_preprocessed_readmission.pkl` and `mimic_preprocessed_mortality.pkl`.

## Scripts

- Create the folders `./checkpoints`, and `./results`.
- To run baselines, refer to the corresponding scripts in the `scripts/` folder.

## Acknowledgements

We thank the authors of all baselines. Most of our implementations follow the corresponding original released versions. We gratefully acknowledge the help of Zhenbang Wu in the preprocessing of MIMIC-IV dataset.
