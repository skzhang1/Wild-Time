from wildtime  import baseline_trainer
import argparse
from flaml import AutoML, CFO, tune
from collections import defaultdict
import pandas as pd
import pickle
import itertools
import sys
from csv_recorder import CSVRecorder
import os
import numpy as np
from ray import tune as raytune
import sys
from flaml.data import load_openml_dataset
import time
import random
import arff
import warnings
from torch.utils.data import DataLoader
from torch import nn
from functools import partial
import torch
import torch.optim as optim

configs_drug_erm = {'dataset': 'drug', 'regression': False, 'prediction_type': None, 'method': 'erm', 'device': 0, 'random_seed': 1, 'train_update_iter': 5000, 'lr': 5e-05, 'momentum': 0.9, 'weight_decay': 0.0, 'mini_batch_size': 256, 'reduced_train_prop': None, 'eval_fix': True, 'difficulty': False, 'split_time': 2016, 'eval_next_timestamps': 3, 'load_model': False, 'eval_all_timestamps': False, 'K': 1, 'lisa': False, 'lisa_intra_domain': False, 'mixup': False, 'lisa_start_time': 0, 'mix_alpha': 2.0, 'cut_mix': False, 'num_groups': 3, 'group_size': 2, 'non_overlapping': False, 'ewc_lambda': 1.0, 'gamma': 1.0, 'online': False, 'fisher_n': None, 'emp_FI': False, 'buffer_size': 100, 'coral_lambda': 1.0, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'si_c': 0.1, 'epsilon': 0.001, 'ssl_finetune_iter': 0, 'data_dir': './Data', 'log_dir': './checkpoints', 'results_dir': './results', 'num_workers': 0}
configs_drug_ft = {'dataset': 'drug', 'regression': False, 'prediction_type': None, 'method': 'ft', 'device': 0, 'random_seed': 1, 'train_update_iter': 5000, 'lr': 5e-05, 'momentum': 0.9, 'weight_decay': 0.0, 'mini_batch_size': 256, 'reduced_train_prop': None, 'eval_fix': True, 'difficulty': False, 'split_time': 2016, 'eval_next_timestamps': 3, 'load_model': False, 'eval_all_timestamps': False, 'K': 1, 'lisa': False, 'lisa_intra_domain': False, 'mixup': False, 'lisa_start_time': 0, 'mix_alpha': 2.0, 'cut_mix': False, 'num_groups': 3, 'group_size': 2, 'non_overlapping': False, 'ewc_lambda': 1.0, 'gamma': 1.0, 'online': False, 'fisher_n': None, 'emp_FI': False, 'buffer_size': 100, 'coral_lambda': 1.0, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'si_c': 0.1, 'epsilon': 0.001, 'ssl_finetune_iter': 0, 'data_dir': './Data', 'log_dir': './checkpoints', 'results_dir': './results', 'num_workers': 0}
configs_drug_swa = {'dataset': 'drug', 'regression': False, 'prediction_type': None, 'method': 'swa', 'device': 0, 'random_seed': 1, 'train_update_iter': 5000, 'lr': 5e-05, 'momentum': 0.9, 'weight_decay': 0.0, 'mini_batch_size': 256, 'reduced_train_prop': None, 'eval_fix': True, 'difficulty': False, 'split_time': 2016, 'eval_next_timestamps': 3, 'load_model': False, 'eval_all_timestamps': False, 'K': 1, 'lisa': False, 'lisa_intra_domain': False, 'mixup': False, 'lisa_start_time': 0, 'mix_alpha': 2.0, 'cut_mix': False, 'num_groups': 3, 'group_size': 2, 'non_overlapping': False, 'ewc_lambda': 1.0, 'gamma': 1.0, 'online': False, 'fisher_n': None, 'emp_FI': False, 'buffer_size': 100, 'coral_lambda': 1.0, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'si_c': 0.1, 'epsilon': 0.001, 'ssl_finetune_iter': 0, 'data_dir': './Data', 'log_dir': './checkpoints', 'results_dir': './results', 'num_workers': 0}


def merge_config(method_config, arc_config):
    if arc_config is not None:
        method_config["lr"] = arc_config["lr"]
        method_config["train_update_iter"] = arc_config["iteration_limit"]
    method_config["use_config"] = arc_config
    return method_config 

    
# arc_config = None
parser = argparse.ArgumentParser()
parser.add_argument("--hpo_method", type = str, default = "cfo")
parser.add_argument("--train_type", type = int, default = 0)
parser.add_argument("--robust_method" , type=str, default="erm")
parser.add_argument("--seed" , type=int, default=1)
parser.add_argument("--device" , type=int, default=0)
parser.add_argument("--budget" , type=int, default=10800)


args = parser.parse_args()

seed = args.seed
device = args.device
robust_method = args.robust_method
hpo_method = args.hpo_method
train_type = args.train_type
budget = args.budget
tolerance = 0.01

folder_name = f'./out/drug_{robust_method}_budget_{budget}'
path = f"{folder_name}/{hpo_method}_out/seed_{seed}/"
if not os.path.isdir(path):
    os.makedirs(path)

# logpath = open(os.path.join(path, "std.log"), "w")
# sys.stdout = logpath
# sys.stderr = logpath

def evaluate_function(args_config, configuration):

    seed = args.seed
    device = args.device
    robust_method = args.robust_method
    train_type = args.train_type
    budget = args.budget

    robust_method = eval(robust_method)
    config = merge_config(robust_method, configuration)

    if "seed" in args_config.keys():
        config["random_seed"] = args_config["seed"]
    else:
        config["random_seed"] = seed

    config['device'] = device
    config["train_type"] = train_type
    configs = argparse.Namespace(**config)
    final_return = baseline_trainer.train(configs)

    return {"val_loss":np.mean(final_return), "std": np.std(final_return), "worst": np.max(final_return), "folds": final_return}
    

# search_space = {
#     "lr": raytune.loguniform(1e-4, 1e-1),
#     "batch_size":raytune.choice([32,64,128,256]),
#     "iteration_limit": raytune.randint(3000, 5000),
#     'n_conv_channels_c1' :  raytune.qlograndint(16, 512, q= 2, base = 2),
#     'kernel_size_c1' :  raytune.randint(2, 5),
#     'has_max_pool_c1':  raytune.choice([0,1]),
#     'n_conv_channels_c2' :  raytune.qlograndint(16, 512, q= 2, base = 2),
#     'kernel_size_c2' :  raytune.randint(2, 5),
#     'has_max_pool_c2':  raytune.choice([0,1]),
#     'n_conv_channels_c3' :  raytune.qlograndint(16, 512, q= 2, base = 2),
#     'kernel_size_c3' :  raytune.randint(2, 5),
#     'has_max_pool_c3':  raytune.choice([0,1]),
#     'n_conv_channels_c4' :  raytune.qlograndint(16, 512, q= 2, base = 2),
#     'kernel_size_c4' :  raytune.randint(2, 5),
#     'has_max_pool_c4':  raytune.choice([0,1]),
# }

# low_cost_partial_config = {
#     "lr": 1e-1,
#     "batch_size": 32,
#     "iteration_limit": 3000,

#     'n_conv_channels_c1' :  16,
#     'kernel_size_c1' :  3,
#     'n_conv_channels_c2' :  16,
#     'kernel_size_c2' :  3,
#     'n_conv_channels_c3' :  16,
#     'kernel_size_c3' :  3,
#     'n_conv_channels_c4' :  16,
#     'kernel_size_c4' :  3,

#     'has_max_pool_c1':  1,
#     'has_max_pool_c2':  1,
#     'has_max_pool_c3':  1,
#     'has_max_pool_c4':  1,
# }
# points_to_evaluate = None 

# if hpo_method == "lexico_var":
#     lexico_objectives = {}
#     lexico_objectives["metrics"] = ["val_loss", "std"]
#     lexico_objectives["tolerances"] = {"val_loss": 0.01, "std": 0.0} # check
#     lexico_objectives["targets"] = {"val_loss": 0.0, "std": 0.0}
#     lexico_objectives["modes"] = ["min", "min"]
# elif hpo_method == "lexico_worst":
#     lexico_objectives = {}
#     lexico_objectives["metrics"] = ["val_loss", "worst"]
#     lexico_objectives["tolerances"] = {"val_loss": 0.01, "worst": 0.0} # check
#     lexico_objectives["targets"] = {"val_loss": 0.0, "worst": 0.0}
#     lexico_objectives["modes"] = ["min", "min"]
# else:
#     lexico_objectives = None

# if hpo_method in ["lexico_var", "lexico_worst"]:
#     if tolerance is None:
#         raise ValueError('Have to set tolerance for lexico.')
#         exit(1)
#     analysis = tune.run(
#         evaluation_function = partial(evaluate_function,args),
#         num_samples=-1,
#         time_budget_s=budget,
#         config=search_space,
#         use_ray=False,
#         lexico_objectives=lexico_objectives,
#         low_cost_partial_config=low_cost_partial_config,
#         points_to_evaluate=points_to_evaluate,
#         local_dir=path,
#         verbose=3,
#     )
# else:
#     algo = CFO(
#         space=search_space,
#         metric="val_loss",
#         mode="min",
#         seed=seed,
#         low_cost_partial_config=low_cost_partial_config,
#         points_to_evaluate=points_to_evaluate,
#     )
#     analysis = tune.run(
#         evaluation_function = partial(evaluate_function,args),
#         time_budget_s=budget,
#         search_alg=algo,
#         use_ray=False,
#         num_samples=-1,
#         metric="val_loss",
#         local_dir=path,
#         verbose=3,
#     )

# #----------save results-------------
# resul_info = {}
# resul_info["best_result"] = analysis.best_result
# resul_info["best_config"] = analysis.best_config


# # save to /out/drug_xgboost_rmse_b1800/valid.csv
# from csv_recorder import CSVRecorder
# recorder = CSVRecorder(
#     folder_name=folder_name,
#     algorithm=hpo_method,
#     csv_name='valid',  # set to 'test' when testing
#     seed=seed,
#     tolerance=0.01,
# )
# recorder.add_result(resul_info["best_result"])
# recorder.save_to_csv()

# # save best_result and best_config
# savepath = os.path.join(path, "result_info.pckl")
# f = open(savepath, "wb")
# pickle.dump(resul_info, f)
# f.close()
# print("best_result", analysis.best_result)
# print("best_config", analysis.best_config)

# logpath.close()