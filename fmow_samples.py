import pickle
import os
from data.utils import Mode
import numpy as np
import torch

def get_label_prop(dataset_name, num_classes, datasets):
    label_prop = {}
    for t in datasets.keys():
        label_prop[t] = {label: 0 for label in range(num_classes)}
        for label in range(num_classes):
            # if dataset_name in ['yearbook', 'precipitation', 'mimic']:
            #     num_samples_t = len(np.array(datasets[t][Mode.TEST_OOD]['labels']) == label)
            #     prop = num_samples_t / len(np.array(datasets[t][Mode.TEST_OOD]['labels']))
            if dataset_name in ['fmow', 'yearbook', 'precipitation', 'mimic']:
                labels_t = datasets[t][Mode.TEST_OOD]['labels']
                if torch.is_tensor(labels_t):
                    labels_t = labels_t.detach().cpu().numpy()
                else:
                    labels_t = np.array(labels_t)
                num_samples_t = np.array(labels_t == label).sum()
                prop = num_samples_t / len(labels_t)
            elif dataset_name in ['arxiv', 'huffpost']:
                num_samples_t = len(datasets[t][Mode.TEST_OOD]['category'] == label)
                prop = num_samples_t / len(datasets[t][Mode.TEST_OOD]['category'])
            label_prop[t][label] = round(100 * prop, 2)
    return label_prop

def get_label_count(dataset_name, num_classes, datasets):
    label_count = {i: [] for i in range(num_classes)}
    for label in range(num_classes):
        for t in datasets.keys():
            if dataset_name in ['fmow', 'yearbook', 'precipitation']:
                labels_t = datasets[t][Mode.TEST_OOD]['labels']
                if torch.is_tensor(labels_t):
                    labels_t = labels_t.detach().cpu().numpy()
                else:
                    labels_t = np.array(labels_t)
                num_samples_t = np.array(labels_t == label).sum()
            elif dataset_name in ['mimic']:
                print(label)
                print(datasets[t][Mode.TEST_OOD]['labels'].astype(np.int))
                num_samples_t = np.nonzero(datasets[t][Mode.TEST_OOD]['labels'].astype(np.int) == label)[0].sum()
                print(t, num_samples_t)
            elif dataset_name in ['arxiv', 'huffpost']:
                labels_t = np.array(datasets[t][Mode.TEST_OOD]['category'])
                num_samples_t = np.array(labels_t == label).sum()
            label_count[label].append(num_samples_t)
    return label_count


if __name__ == '__main__':
    # datasets = pickle.load(open(os.path.join('./Data', 'fmow.pkl'), 'rb'))
    # label_count = get_label_count('fmow', 62, datasets)
    # print(label_count)
    # datasets = pickle.load(open(os.path.join('./Data', 'huffpost.pkl'), 'rb'))
    # label_count = get_label_count('huffpost', 11, datasets)
    # print(label_count)
    # datasets = pickle.load(open(os.path.join('./Data', 'yearbook.pkl'), 'rb'))
    # label_count = get_label_count('yearbook', 2, datasets)
    # print(label_count)
    # datasets = pickle.load(open(os.path.join('./Data', 'arxiv.pkl'), 'rb'))
    # label_count = get_label_count('arxiv', 172, datasets)
    # print(label_count)
    # datasets = pickle.load(open(os.path.join('./Data', 'precipitation.pkl'), 'rb'))
    # label_count = get_label_count('precipitation', 9, datasets)
    # print(label_count)
    datasets = pickle.load(open(os.path.join('./Data', 'mimic_preprocessed_mortality.pkl'), 'rb'))
    label_count = get_label_count('mimic', 2, datasets)
    print(label_count)
    print()
    datasets = pickle.load(open(os.path.join('./Data', 'mimic_preprocessed_readmission.pkl'), 'rb'))
    label_count = get_label_count('mimic', 2, datasets)
    print(label_count)