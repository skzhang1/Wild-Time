# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted by TDC.

import os
import pickle

import numpy as np

from data.MIMIC.get_stay_dict import get_stay_dict
from data.utils import Mode

ID_HELD_OUT = 0.2

def preprocess_reduced_train_set(args):
    print(f'Preprocessing reduced train proportion dataset and saving to mimic_preprocessed_{args.prediction_type}_{args.reduced_train_prop}.pkl')
    np.random.seed(0)

    orig_data_file = os.path.join(args.data_dir, f'mimic_preprocessed_{args.prediction_type}.pkl')
    dataset = pickle.load(open(orig_data_file, 'rb'))
    years = list(sorted(dataset.keys()))
    train_fraction = args.reduced_train_prop / (1 - ID_HELD_OUT)

    for year in years:
        train_codes = dataset[year][Mode.TRAIN]['code']
        train_labels = dataset[year][Mode.TRAIN]['labels']

        num_train_samples = len(train_labels)
        reduced_num_train_samples = int(train_fraction * num_train_samples)
        idxs = np.random.permutation(np.arange(num_train_samples))
        train_idxs = idxs[:reduced_num_train_samples].astype(int)

        new_train_codes = np.array(train_codes)[train_idxs]
        new_train_labels = np.array(train_labels)[train_idxs]
        dataset[year][Mode.TRAIN]['code'] = np.stack(new_train_codes, axis=0)
        dataset[year][Mode.TRAIN]['labels'] = np.array(new_train_labels)

    preprocessed_data_file = os.path.join(args.data_dir, f'mimic_preprocessed_{args.prediction_type}_{args.reduced_train_prop}.pkl')
    pickle.dump(dataset, open(preprocessed_data_file, 'wb'))
    np.random.seed(args.random_seed)


def preprocess_MIMIC(data, type):
    ENV = [i for i in list(range(2008, 2020))]
    datasets = {}
    temp_datasets = {}

    for i in ENV:
        datasets[i] = {}
        temp_datasets[i] = {'code':[], 'labels':[]}

    for eachadmit in data:
        year = int(data[eachadmit].icu_timestamp)
        if year in temp_datasets:
            if type not in temp_datasets[year]:
                temp_datasets[year][type]=[]
            if type == 'mortality':
                temp_datasets[year]['labels'].append(data[eachadmit].mortality)
            elif type == 'readmission':
                temp_datasets[year]['labels'].append(data[eachadmit].readmission)
            dx_list = ['dx' for _ in data[eachadmit].diagnosis]
            tr_list = ['tr' for _ in data[eachadmit].treatment]
            temp_datasets[year]['code'].append([data[eachadmit].diagnosis + data[eachadmit].treatment, dx_list + tr_list])

    for eachyear in temp_datasets.keys():
        temp_datasets[eachyear]['labels'] = np.array(temp_datasets[eachyear]['labels'])
        temp_datasets[eachyear]['code'] = np.array(temp_datasets[eachyear]['code'])
        num_samples = temp_datasets[eachyear]['labels'].shape[0]
        seed_ = np.random.get_state()
        np.random.seed(0)
        idxs = np.random.permutation(np.arange(num_samples))
        np.random.set_state(seed_)
        num_train_samples = int((1 - ID_HELD_OUT) * num_samples)
        datasets[eachyear][Mode.TRAIN] = {}
        datasets[eachyear][Mode.TRAIN]['code'] = temp_datasets[eachyear]['code'][idxs[:num_train_samples]]
        datasets[eachyear][Mode.TRAIN]['labels'] = temp_datasets[eachyear]['labels'][idxs[:num_train_samples]]

        datasets[eachyear][Mode.TEST_ID] = {}
        datasets[eachyear][Mode.TEST_ID]['code'] = temp_datasets[eachyear]['code'][idxs[num_train_samples:]]
        datasets[eachyear][Mode.TEST_ID]['labels'] = temp_datasets[eachyear]['labels'][idxs[num_train_samples:]]

        datasets[eachyear][Mode.TEST_OOD] = {}
        datasets[eachyear][Mode.TEST_OOD]['code'] = temp_datasets[eachyear]['code']
        datasets[eachyear][Mode.TEST_OOD]['labels'] = temp_datasets[eachyear]['labels']

    with open(f'./Data/mimic_preprocessed_{type}.pkl','wb') as f:
        pickle.dump(datasets, f)


def preprocess_orig(args):
    if not os.path.exists('./Data/mimic_stay_dict.pkl'):
        get_stay_dict()
    data = pickle.load(open('./Data/mimic_stay_dict.pkl', 'rb'))
    preprocess_MIMIC(data, 'readmission')
    preprocess_MIMIC(data, 'mortality')


def preprocess(args):
    np.random.seed(0)
    if not os.path.isfile(os.path.join(args.data_dir, f'mimic_preprocessed_{args.prediction_type}.pkl')):
        preprocess_orig(args)
    if args.reduced_train_prop is not None:
        if not os.path.isfile(os.path.join(args.data_dir, f'mimic_preprocessed_{args.prediction_type}_{args.reduced_train_prop}.pkl')):
            preprocess_reduced_train_set(args)
    np.random.seed(args.random_seed)