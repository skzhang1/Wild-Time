from wildtime  import baseline_trainer
import argparse

erm = {'dataset': 'yearbook', 'regression': False, 'prediction_type': None, 'method': 'erm', 'device': 0, 'random_seed': 1, 'train_update_iter': 3000, 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0, 'mini_batch_size': 32, 'reduced_train_prop': None, 'eval_fix': True, 'difficulty': False, 'split_time': 1970, 'eval_next_timestamps': 10, 'load_model': False, 'eval_all_timestamps': False, 'K': 1, 'lisa': False, 'lisa_intra_domain': False, 'mixup': False, 'lisa_start_time': 0, 'mix_alpha': 2.0, 'cut_mix': False, 'num_groups': 10, 'group_size': 5, 'non_overlapping': False, 'ewc_lambda': 1.0, 'gamma': 1.0, 'online': False, 'fisher_n': None, 'emp_FI': False, 'buffer_size': 100, 'coral_lambda': 1.0, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'si_c': 0.1, 'epsilon': 0.001, 'ssl_finetune_iter': 300, 'data_dir': './Data', 'log_dir': './checkpoints', 'results_dir': './results', 'num_workers': 0}
lisa = {'dataset': 'yearbook', 'regression': False, 'prediction_type': None, 'method': 'erm', 'device': 0, 'random_seed': 1, 'train_update_iter': 3000, 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0, 'mini_batch_size': 32, 'reduced_train_prop': None, 'eval_fix': True, 'difficulty': False, 'split_time': 1970, 'eval_next_timestamps': 10, 'load_model': False, 'eval_all_timestamps': False, 'K': 1, 'lisa': True, 'lisa_intra_domain': False, 'mixup': False, 'lisa_start_time': 0, 'mix_alpha': 2.0, 'cut_mix': False, 'num_groups': 10, 'group_size': 5, 'non_overlapping': False, 'ewc_lambda': 1.0, 'gamma': 1.0, 'online': False, 'fisher_n': None, 'emp_FI': False, 'buffer_size': 100, 'coral_lambda': 1.0, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'si_c': 0.1, 'epsilon': 0.001, 'ssl_finetune_iter': 300, 'data_dir': './Data', 'log_dir': './checkpoints', 'results_dir': './results', 'num_workers': 0}
ft = {'dataset': 'yearbook', 'regression': False, 'prediction_type': None, 'method': 'ft', 'device': 0, 'random_seed': 1, 'train_update_iter': 3000, 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0, 'mini_batch_size': 32, 'reduced_train_prop': None, 'eval_fix': True, 'difficulty': False, 'split_time': 1970, 'eval_next_timestamps': 10, 'load_model': False, 'eval_all_timestamps': False, 'K': 1, 'lisa': False, 'lisa_intra_domain': False, 'mixup': False, 'lisa_start_time': 0, 'mix_alpha': 2.0, 'cut_mix': False, 'num_groups': 10, 'group_size': 5, 'non_overlapping': False, 'ewc_lambda': 1.0, 'gamma': 1.0, 'online': False, 'fisher_n': None, 'emp_FI': False, 'buffer_size': 100, 'coral_lambda': 1.0, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'si_c': 0.1, 'epsilon': 0.001, 'ssl_finetune_iter': 300, 'data_dir': './Data', 'log_dir': './checkpoints', 'results_dir': './results', 'num_workers': 0}
simclr = {'dataset': 'yearbook', 'regression': False, 'prediction_type': None, 'method': 'simclr', 'device': 0, 'random_seed': 1, 'train_update_iter': 2700, 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0, 'mini_batch_size': 32, 'reduced_train_prop': None, 'eval_fix': True, 'difficulty': False, 'split_time': 1970, 'eval_next_timestamps': 10, 'load_model': False, 'eval_all_timestamps': False, 'K': 1, 'lisa': False, 'lisa_intra_domain': False, 'mixup': False, 'lisa_start_time': 0, 'mix_alpha': 2.0, 'cut_mix': False, 'num_groups': 10, 'group_size': 5, 'non_overlapping': False, 'ewc_lambda': 1.0, 'gamma': 1.0, 'online': False, 'fisher_n': None, 'emp_FI': False, 'buffer_size': 100, 'coral_lambda': 1.0, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'si_c': 0.1, 'epsilon': 0.001, 'ssl_finetune_iter': 300, 'data_dir': './Data', 'log_dir': './checkpoints', 'results_dir': './results', 'num_workers': 0}
swa = {'dataset': 'yearbook', 'regression': False, 'prediction_type': None, 'method': 'swa', 'device': 0, 'random_seed': 1, 'train_update_iter': 3000, 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0, 'mini_batch_size': 32, 'reduced_train_prop': None, 'eval_fix': True, 'difficulty': False, 'split_time': 1970, 'eval_next_timestamps': 10, 'load_model': False, 'eval_all_timestamps': False, 'K': 1, 'lisa': False, 'lisa_intra_domain': False, 'mixup': False, 'lisa_start_time': 0, 'mix_alpha': 2.0, 'cut_mix': False, 'num_groups': 10, 'group_size': 5, 'non_overlapping': False, 'ewc_lambda': 1.0, 'gamma': 1.0, 'online': False, 'fisher_n': None, 'emp_FI': False, 'buffer_size': 100, 'coral_lambda': 1.0, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'si_c': 0.1, 'epsilon': 0.001, 'ssl_finetune_iter': 300, 'data_dir': './Data', 'log_dir': './checkpoints', 'results_dir': './results', 'num_workers': 0}

def merge_config(method_config, arc_config):
    if arc_config is not None:
        method_config["lr"] = arc_config["lr"]
        method_config["train_update_iter"] = arc_config["iteration_limit"]
    method_config["use_config"] = arc_config
    return method_config 

parser = argparse.ArgumentParser()
parser.add_argument("--robust_method" , type=str, default="erm")
parser.add_argument("--seed" , type=int, default=1)
parser.add_argument("--device" , type=int, default=0)
args = parser.parse_args()

robust_method = args.robust_method
seed = args.seed
device = args.device

robust_method = eval(robust_method)
arc_config = None
config = merge_config(robust_method, arc_config)
config["random_seed"] = seed
config['device'] = device

configs = argparse.Namespace(**config)
baseline_trainer.train(configs)

