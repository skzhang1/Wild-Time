import os
import torch
import warnings
from torchcontrib.optim import SWA as SWA_optimizer
from methods.base_trainer import BaseTrainer
from data.utils import Mode
from dataloaders import InfiniteDataLoader

class SWA(BaseTrainer):
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        base_opt = self.optimizer
        self.optimizer = SWA_optimizer(base_opt)
        self.optimizer.defaults = base_opt.defaults
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def train_offline(self):
        for i, t in enumerate(self.train_dataset.ENV):
            print(i, t)
            if t < self.split_time:
                # Collate data from all time steps 1, ..., m-1
                self.train_dataset.mode = Mode.TRAIN
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.update_historical(i + 1)
                self.train_dataset.mode = Mode.TEST_ID
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.update_historical(i + 1, data_del=True)
            elif t == self.split_time:
                self.train_dataset.mode = Mode.TRAIN
                self.train_dataset.update_current_timestamp(t)
                # Train
                train_id_dataloader = InfiniteDataLoader(
                    dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size, num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                if self.args.load_model:
                    self.load_model(t)
                else:
                    self.train_step(train_id_dataloader)
                    self.save_model(t)
                    self.optimizer.update_swa()
                break
        self.optimizer.swap_swa_sgd()

    def get_optimizer_path(self, timestep):
        model_str = f'{str(self.train_dataset)}_{str(self)}_time={timestep}'
        path = os.path.join(self.args.log_dir, model_str, "_opt")
        return path

    def save_model(self, timestep):
        model_path = self.get_model_path(timestep)
        opt_path = self.get_optimizer_path(timestep)

        torch.save(self.network.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), opt_path)

    def load_model(self, timestep):
        model_path = self.get_model_path(timestep)
        opt_path = self.get_optimizer_path(timestep)

        self.network.load_state_dict(torch.load(model_path), strict=False)
        self.optimizer.load_state_dict(torch.load(opt_path), strict=False)

    def evaluate_online(self):
        end = len(self.eval_dataset.ENV) - self.eval_next_timesteps
        for i, t in enumerate(self.eval_dataset.ENV[:end]):
            from copy import copy
            model_checkpoint = copy.deepcopy(self.network)
            self.load_model(t)
            self.optimizer.swap_swa_sgd()

            avg_acc, worst_acc, best_acc = self.evaluate_stream(i + 1)
            self.task_accuracies[t] = avg_acc
            self.worst_time_accuracies[t] = worst_acc
            self.best_time_accuracies[t] = best_acc

            self.network = model_checkpoint

    def __str__(self):
        return f'SWA-{self.base_trainer_str}'
