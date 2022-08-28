import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from data.MIMIC.preprocess import preprocess
from data.utils import Mode

class MIMICBase(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        preprocess(args)
        if args.reduced_train_prop is None:
            self.data_file = f'{str(self)}_preprocessed.pkl'
        else:
            self.data_file = f'{str(self)}_preprocessed_{args.reduced_train_prop}.pkl'
        self.datasets = pickle.load(open(os.path.join(args.data_dir, self.data_file), 'rb'))

        self.num_classes = 2
        self.current_time = 0
        self.mini_batch_size = args.mini_batch_size
        self.mode = Mode.TRAIN

        self.ENV = list(sorted(self.datasets.keys()))
        self.num_tasks = len(self.ENV)
        self.num_examples = {i: self.datasets[i][self.mode]['labels'].shape[0] for i in self.ENV}

        ## create a datasets object
        self.class_id_list = {i: {} for i in range(self.num_classes)}
        self.input_dim = []
        cumulative_batch_size = 0
        self.task_idxs = {}
        start_idx = 0

        for i in self.ENV:
            end_idx = start_idx + self.datasets[i][self.mode]['labels'].shape[0]
            self.task_idxs[i] = [start_idx, end_idx]
            start_idx = end_idx

            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[i][self.mode]['labels'].astype(np.int) == classid)[0]
                self.class_id_list[classid][i] = sel_idx
            print(f'Year {str(i)} loaded')

            cumulative_batch_size += min(self.mini_batch_size, self.num_examples[i])
            if args.method in ['erm']:
                self.input_dim.append((cumulative_batch_size, 3, 32, 32))
            else:
                self.input_dim.append((min(self.mini_batch_size, self.num_examples[i]), 3, 32, 32))

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['code'] = np.concatenate(
            (self.datasets[time][self.mode]['code'], self.datasets[prev_time][self.mode]['code']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'].astype(np.int) == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_historical_K(self, idx, K):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.window_start = self.ENV[max(0, idx - K)]
        if idx >= K:
            last_K_num_samples = self.input_dim[idx - K][0]
            self.datasets[time][self.mode]['code'] = np.concatenate(
                (self.datasets[time][self.mode]['code'], self.datasets[prev_time][self.mode]['code'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['labels'] = np.concatenate(
                (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels'][:-last_K_num_samples]), axis=0)
            del self.datasets[prev_time]
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'].astype(np.int) == classid)[0]
                self.class_id_list[classid][time] = sel_idx
        else:
            self.update_historical(idx)

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        # time = self.ENV[time_idx]
        idx_all = self.class_id_list[classid][time_idx]
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)
        codes = [item[0] for item in self.datasets[time_idx][self.mode]['code'][sel_idx]]
        types = [item[1] for item in self.datasets[time_idx][self.mode]['code'][sel_idx]]
        code = (codes, types)
        label = self.datasets[time_idx][self.mode]['labels'][sel_idx].astype(np.int)

        return code, torch.LongTensor([label]).squeeze(0).unsqueeze(1).cuda()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return f'mimic_{self.args.prediction_type}'


class MIMIC(MIMICBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        if self.args.difficulty and self.mode == Mode.TRAIN:
            # Pick a time step from all previous timesteps
            idx = self.ENV.index(self.current_time)
            window = np.arange(0, idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time][self.mode]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            index = sel_idx

        code = self.datasets[self.current_time][self.mode]['code'][index]
        label = int(self.datasets[self.current_time][self.mode]['labels'][index])
        label_tensor = torch.LongTensor([label])

        return code, label_tensor

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])


class MIMICGroup(MIMICBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        self.window_end = self.ENV[0]
        self.groupnum = 0

    def __getitem__(self, index):
        if self.mode == Mode.TRAIN:
            np.random.seed(index)
            # Select group ID
            idx = self.ENV.index(self.current_time)
            if self.args.non_overlapping:
                possible_groupids = [i for i in range(0, max(1, idx - self.group_size + 1), self.group_size)]
                if len(possible_groupids) == 0:
                    possible_groupids = [np.random.randint(self.group_size)]
            else:
                possible_groupids = [i for i in range(max(1, idx - self.group_size + 1))]
            groupid = np.random.choice(possible_groupids)

            # Pick a time step in the sliding window
            window = np.arange(max(0, idx - groupid - self.group_size), idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx = self.task_idxs[sel_time][0]
            end_idx = self.task_idxs[sel_time][1]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            code = self.datasets[self.current_time][self.mode]['code'][sel_idx]
            label = int(self.datasets[self.current_time][self.mode]['labels'][sel_idx])
            label_tensor = torch.LongTensor([label])
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx

            return code, label_tensor, group_tensor

        else:
            code = self.datasets[self.current_time][self.mode]['code'][index]
            label = int(self.datasets[self.current_time][self.mode]['labels'][index])
            label_tensor = torch.LongTensor([label])

            return code, label_tensor

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])
