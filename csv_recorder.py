import pandas as pd
import os 
import json
import numpy as np

class CSVRecorder:
    def __init__(self, folder_name, algorithm, csv_name, seed, tolerance) -> None:
        folder = f'{folder_name}/{algorithm}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.csv_path = f'{folder}/{csv_name}.csv'
        
        self.seed = seed
        self.tolerance = tolerance
        self.csv_dict = {'seed': [],
                        'tolerance': [],    
                        'folds': [],
                        'val_loss': [],
                        'std': [],
                        'worst': [],     
                        }

    def add_result(self, result):
        # keys = ['folds', 'val_loss', 'std', 'worst']
        keys = ['folds', 'val_loss', 'std', 'worst']
        for k in keys:
            self.csv_dict[k].append(result[k])
        self.csv_dict['seed'].append(self.seed)
        self.csv_dict['tolerance'].append(self.tolerance)

    
    def add_each_seed(self, result, seed, tolerance):
        # keys = ['folds', 'val_loss', 'std', 'worst']
        keys = ['folds', 'val_loss', 'std', 'worst']
        for k in keys:
            self.csv_dict[k].append(result[k])
        self.csv_dict['seed'].append(seed)
        self.csv_dict['tolerance'].append(tolerance)
        

    def save_to_csv(self):
        for k in self.csv_dict.keys():
            print(k, self.csv_dict[k])

        new_data = pd.DataFrame(self.csv_dict)
        if os.path.exists(self.csv_path):
            data = pd.read_csv(self.csv_path)
            data = pd.concat([data, new_data])
        else:
            data = new_data
        
        data.to_csv(self.csv_path, index=False)
        print(f'Saving to {self.csv_path} succesfully.')

    def check_entry_exist(self,):
        if not os.path.exists(self.csv_path):
            return False
        csv_file = pd.read_csv(self.csv_path)

        with_seed = csv_file[csv_file['seed'] == self.setting['seed']]
        with_seed_tolerance = with_seed[csv_file['tolerance'] == self.setting['tolerance']]
        if len(with_seed_tolerance) > 0:
            print(with_seed_tolerance)
            print(f'The entry already exists.')
            return True
        return False


class CSV_Analyser:
    def __init__(self, dataset_folder, best_tolerance) -> None:
        dir = os.listdir(dataset_folder)
        print(dir)
        self.algorithms = ['cfo', 'lexico_var', 'lexico_worst']
        self.splits = ['valid',] # 'test'
        for a in self.algorithms:
            assert a in dir, f'Folder {a} is not in {dataset_folder}. Please check folder.'

        self.dataset_folder = dataset_folder
        self.best_tolerance = best_tolerance
    

    def read_all(self,):
        all_entries=[]
        for s in self.splits:
            for a in self.algorithms:
                file = os.path.join(self.dataset_folder, a, s+'.csv')
                entries = pd.read_csv(file)
                if a != 'cfo':
                    entries = entries[entries['tolerance'] == self.best_tolerance]
                all_entries.append(entries)
        self.all_entries = all_entries


    def get_worst_loss_allSeed_perFold(self):
        pass
    
    def get_average_loss_allSedd_perFold(self):
        pass

