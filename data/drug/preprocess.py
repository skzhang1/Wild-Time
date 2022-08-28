# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted by TDC.

import os
import pickle

import numpy as np
import pandas as pd
from tdc import BenchmarkGroup

from data.utils import Mode

ID_HELD_OUT = 0.2


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
              'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
               '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
               'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
               'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
               'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

MAX_SEQ_PROTEIN = 1000
MAX_SEQ_DRUG = 100


def trans_protein(x):
    temp = list(x.upper())
    temp = [i if i in amino_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_PROTEIN:
        temp = temp + ['?'] * (MAX_SEQ_PROTEIN - len(temp))
    else:
        temp = temp[:MAX_SEQ_PROTEIN]
    return temp


def trans_drug(x):
    temp = list(x)
    temp = [i if i in smiles_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_DRUG:
        temp = temp + ['?'] * (MAX_SEQ_DRUG - len(temp))
    else:
        temp = temp[:MAX_SEQ_DRUG]
    return temp


def preprocess_reduced_train_set(args):
    print(f'Preprocessing reduced train proportion dataset and saving to drug_preprocessed_{args.reduced_train_prop}.pkl')

    orig_data_file = os.path.join(args.data_dir, f'drug_preprocessed.pkl')
    dataset = pickle.load(open(orig_data_file, 'rb'))
    years = [year for year in list(range(2013, 2021))]
    train_fraction = args.reduced_train_prop / (1 - ID_HELD_OUT)

    start_idx = 0
    end_idx = 0
    task_idxs = {}
    for year in years:
        train_data = dataset[year][Mode.TRAIN]
        if year != 2019:
            start_idx = end_idx
            end_idx = start_idx + len(train_data)
        elif year == 2019:
            start_idx = 0
            end_idx = len(train_data)
        task_idxs[year] = [start_idx, end_idx]

        num_train_samples = len(train_data)
        reduced_num_train_samples = int(train_fraction * num_train_samples)

        new_train_data = train_data.loc[:reduced_num_train_samples, :].reset_index(drop=True)
        dataset[year][Mode.TRAIN] = new_train_data

    preprocessed_data_file = os.path.join(args.data_dir, f'drug_preprocessed_{args.reduced_train_prop}.pkl')
    with open(preprocessed_data_file, 'wb') as f:
        pickle.dump(dataset, f)


def preprocess_orig(args):
    group = BenchmarkGroup(name='DTI_DG_Group', path=os.path.join(args.data_dir, 'TDC_OOD'))
    benchmark = group.get('BindingDB_Patent')
    train_val, test, name = benchmark['train_val'], benchmark['test'], benchmark['name']

    unique_drug = pd.Series(train_val['Drug'].unique()).apply(trans_drug)
    unique_dict_drug = dict(zip(train_val['Drug'].unique(), unique_drug))
    train_val['Drug_Enc'] = [unique_dict_drug[str(i)] for i in train_val['Drug']]

    unique_target = pd.Series(train_val['Target'].unique()).apply(trans_protein)
    unique_dict_target = dict(zip(train_val['Target'].unique(), unique_target))
    train_val['Target_Enc'] = [unique_dict_target[str(i)] for i in train_val['Target']]

    unique_drug = pd.Series(test['Drug'].unique()).apply(trans_drug)
    unique_dict_drug = dict(zip(test['Drug'].unique(), unique_drug))
    test['Drug_Enc'] = [unique_dict_drug[str(i)] for i in test['Drug']]

    unique_target = pd.Series(test['Target'].unique()).apply(trans_protein)
    unique_dict_target = dict(zip(test['Target'].unique(), unique_target))
    test['Target_Enc'] = [unique_dict_target[str(i)] for i in test['Target']]

    datasets = {}
    task_idxs = {}
    ENV = [year for year in list(range(2013, 2021))]
    start_idx = 0
    end_idx = 0
    for year in ENV:
        datasets[year] = {}
        if year < 2019:
            df_data = train_val[train_val.Year == year]
        else:
            df_data = test[test.Year == year]

        if year != 2019:
            start_idx = end_idx
            end_idx = start_idx + len(df_data)
        elif year == 2019:
            start_idx = 0
            end_idx = len(df_data)
        task_idxs[year] = [start_idx, end_idx]

        num_samples = len(df_data)
        seed_ = np.random.get_state()
        np.random.seed(0)
        idxs = np.random.permutation(np.arange(start_idx, end_idx))
        np.random.set_state(seed_)
        num_train_samples = int((1 - ID_HELD_OUT) * num_samples)
        datasets[year][Mode.TRAIN] = df_data.loc[idxs[:num_train_samples], :].reset_index()
        datasets[year][Mode.TEST_ID] = df_data.loc[idxs[num_train_samples:], :].reset_index()
        datasets[year][Mode.TEST_OOD] = df_data.reset_index()

    with open('drug_preprocessed.pkl', 'wb') as f:
        pickle.dump(datasets, f)


def preprocess(args):
    np.random.seed(0)
    if not os.path.isfile(os.path.join(args.data_dir, 'drug_preprocessed.pkl')):
        preprocess_orig(args)
    if args.reduced_train_prop is not None:
        if not os.path.isfile(os.path.join(args.data_dir, f'drug_preprocessed_{args.reduced_train_prop}.pkl')):
            preprocess_reduced_train_set(args)
    np.random.seed(args.random_seed)