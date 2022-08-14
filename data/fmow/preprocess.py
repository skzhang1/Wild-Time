from collections import defaultdict

import os
import numpy as np
import pickle
import torch
from wilds import get_dataset

from data.utils import Mode

ID_HELD_OUT = 0.1

def preprocess_reduced_train_set(args):
    print(f'Preprocessing reduced train proportion dataset and saving to fmow_{args.reduced_train_prop}.pkl')
    np.random.seed(0)

    orig_data_file = os.path.join(args.data_dir, f'fmow.pkl')
    dataset = pickle.load(open(orig_data_file, 'rb'))
    years = list(sorted(dataset.keys()))
    train_fraction = args.reduced_train_prop / (1 - ID_HELD_OUT)

    for year in years:
        train_image_idxs = dataset[year][Mode.TRAIN]['image_idxs']
        train_labels = dataset[year][Mode.TRAIN]['labels']

        num_train_samples = len(train_labels)
        reduced_num_train_samples = int(train_fraction * num_train_samples)
        idxs = np.random.permutation(np.arange(num_train_samples))
        train_idxs = idxs[:reduced_num_train_samples].astype(int)

        new_train_image_idxs = np.array(train_image_idxs)[train_idxs]
        new_train_labels = np.array(train_labels)[train_idxs]
        dataset[year][Mode.TRAIN]['image_idxs'] = np.stack(new_train_image_idxs, axis=0)
        dataset[year][Mode.TRAIN]['labels'] = np.array(new_train_labels)

    preprocessed_data_file = os.path.join(args.data_dir, f'fmow_{args.reduced_train_prop}.pkl')
    pickle.dump(dataset, open(preprocessed_data_file, 'wb'))
    np.random.seed(args.random_seed)


def get_image_idxs_and_labels(split: str, data_dir: str):
    dataset = get_dataset(dataset="fmow", root_dir=data_dir, download=True)
    split_array = dataset.split_array
    split_dict = dataset.split_dict

    y_array = []

    split_mask = split_array == split_dict[split]
    split_idx = np.where(split_mask)[0]
    y_array.append(dataset.y_array[split_idx])
    years = dataset.metadata_array[split_idx, 1]
    split_unique_years = torch.unique(years).detach().numpy().tolist()
    print(split, split_unique_years)

    image_idxs = defaultdict(list)
    labels = defaultdict(list)
    for year in split_unique_years:
        image_idxs[year].append(dataset.full_idxs[split_idx][torch.where(years == year)])
        labels[year].append(dataset.y_array[split_idx][torch.where(years == year)])

    return image_idxs, labels


def get_train_test_split(image_idxs_year, labels_year):
    num_samples = len(labels_year)
    num_train_samples = int((1 - ID_HELD_OUT) * num_samples)
    seed_ = np.random.get_state()
    np.random.seed(0)
    idxs = np.random.permutation(np.arange(num_samples))
    np.random.set_state(seed_)
    train_image_idxs = image_idxs_year[idxs[:num_train_samples]]
    train_labels = labels_year[idxs[:num_train_samples]]
    test_image_idxs = image_idxs_year[idxs[num_train_samples:]]
    test_labels = labels_year[idxs[num_train_samples:]]

    print(type(train_image_idxs), type(train_labels))

    return train_image_idxs, train_labels, test_image_idxs, test_labels


def preprocess_orig(args):
    # FMoW Split Setup
    # train: 2002 - 2013
    # val_id: 2002 - 2013
    # test_id: 2002 - 2013
    # val_ood: 2013 - 2016
    # test_ood: 2016 - 2018
    # self._split_names = {'train': 'Train', 'id_val': 'ID Val', 'id_test': 'ID Test', 'val': 'OOD Val', 'test': 'OOD Test'}
    datasets = {}
    train_image_idxs, train_labels = get_image_idxs_and_labels('train', args.data_dir)
    val_image_idxs, val_labels = get_image_idxs_and_labels('id_val', args.data_dir)
    test_id_image_idxs, test_id_labels = get_image_idxs_and_labels('id_test', args.data_dir)

    # ID Years 2002 - 2013
    for year in range(0, 11):
        datasets[year] = {}
        datasets[year][Mode.TRAIN] = {}
        datasets[year][Mode.TEST_ID] = {}
        datasets[year][Mode.TEST_OOD] = {}

        datasets[year][Mode.TRAIN]['image_idxs'] = np.concatenate((train_image_idxs[year][0], val_image_idxs[year][0]), axis=0)
        datasets[year][Mode.TRAIN]['labels'] = np.concatenate((train_labels[year][0], val_labels[year][0]), axis=0)
        datasets[year][Mode.TEST_ID]['image_idxs'] = np.array(test_id_image_idxs[year][0])
        datasets[year][Mode.TEST_ID]['labels'] = np.array(test_id_labels[year][0])
        datasets[year][Mode.TEST_OOD]['image_idxs'] = np.concatenate((datasets[year][Mode.TRAIN]['image_idxs'], datasets[year][Mode.TEST_ID]['image_idxs']), axis=0)
        datasets[year][Mode.TEST_OOD]['labels'] = np.concatenate((datasets[year][Mode.TRAIN]['labels'], datasets[year][Mode.TEST_ID]['labels']), axis=0)
    del train_image_idxs, train_labels, val_image_idxs, val_labels, test_id_image_idxs, test_id_labels

    # Intermediate Years 2013 - 2016
    val_ood_image_idxs, val_ood_labels = get_image_idxs_and_labels('val', args.data_dir)
    for year in range(11, 14):
        datasets[year] = {}
        datasets[year][Mode.TRAIN] = {}
        datasets[year][Mode.TEST_ID] = {}
        datasets[year][Mode.TEST_OOD] = {}

        train_image_idxs, train_labels, test_image_idxs, test_labels = get_train_test_split(val_ood_image_idxs[year][0], val_ood_labels[year][0])
        datasets[year][Mode.TRAIN]['image_idxs'] = train_image_idxs
        datasets[year][Mode.TRAIN]['labels'] = train_labels
        datasets[year][Mode.TEST_ID]['image_idxs'] = test_image_idxs
        datasets[year][Mode.TEST_ID]['labels'] = test_labels
        datasets[year][Mode.TEST_OOD]['image_idxs'] = val_ood_image_idxs[year][0]
        datasets[year][Mode.TEST_OOD]['labels'] = val_ood_labels[year][0]
        del train_image_idxs, train_labels, test_image_idxs, test_labels
    del val_ood_image_idxs, val_ood_labels

    # OOD Years 2016 - 2018
    test_ood_image_idxs, test_ood_labels = get_image_idxs_and_labels('test', args.data_dir)
    for year in range(14, 16):
        datasets[year] = {}
        datasets[year][Mode.TRAIN] = {}
        datasets[year][Mode.TEST_ID] = {}
        datasets[year][Mode.TEST_OOD] = {}

        train_image_idxs, train_labels, test_image_idxs, test_labels = get_train_test_split(test_ood_image_idxs[year][0], test_ood_labels[year][0])
        datasets[year][Mode.TRAIN]['image_idxs'] = train_image_idxs
        datasets[year][Mode.TRAIN]['labels'] = train_labels
        datasets[year][Mode.TEST_ID]['image_idxs'] = test_image_idxs
        datasets[year][Mode.TEST_ID]['labels'] = test_labels
        datasets[year][Mode.TEST_OOD]['image_idxs'] = test_ood_image_idxs[year][0]
        datasets[year][Mode.TEST_OOD]['labels'] = test_ood_labels[year][0]
        del train_image_idxs, train_labels, test_image_idxs, test_labels
    del test_ood_image_idxs, test_ood_labels

    print(datasets)

    preprocessed_data_path = os.path.join(args.data_dir, 'fmow.pkl')
    pickle.dump(datasets, open(preprocessed_data_path, 'wb'))

def preprocess(args):
    np.random.seed(0)
    if not os.path.isfile(os.path.join(args.data_dir, 'fmow.pkl')):
        preprocess_orig(args)
    if args.reduced_train_prop is not None:
        if not os.path.isfile(os.path.join(args.data_dir, f'fmow_{args.reduced_train_prop}.pkl')):
            preprocess_reduced_train_set(args)
    np.random.seed(args.random_seed)