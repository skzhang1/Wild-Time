from wildtime import dataloader
import argparse

configs = {'dataset': 'drug', 'regression': False, 'prediction_type': None, 'method': 'erm', 'device': 0, 'random_seed': 1, 'train_update_iter': 5000, 'lr': 5e-05, 'momentum': 0.9, 'weight_decay': 0.0, 'mini_batch_size': 256, 'reduced_train_prop': None, 'eval_fix': True, 'difficulty': False, 'split_time': 2016, 'eval_next_timestamps': 3, 'load_model': False, 'eval_all_timestamps': False, 'K': 1, 'lisa': False, 'lisa_intra_domain': False, 'mixup': False, 'lisa_start_time': 0, 'mix_alpha': 2.0, 'cut_mix': False, 'num_groups': 3, 'group_size': 2, 'non_overlapping': False, 'ewc_lambda': 1.0, 'gamma': 1.0, 'online': False, 'fisher_n': None, 'emp_FI': False, 'buffer_size': 100, 'coral_lambda': 1.0, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'si_c': 0.1, 'epsilon': 0.001, 'ssl_finetune_iter': 0, 'data_dir': './Data', 'log_dir': './checkpoints', 'results_dir': './results', 'num_workers': 0}

configs = argparse.Namespace(**configs)
data = dataloader.getdata(configs)



