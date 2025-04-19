import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR_PATH = '../CMAPSSData/'
BATCH_SIZE = 1


class FD001_dataset(Dataset):
    
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    drop_labels = drop_sensors + ['setting_3']# + setting_names
    
    train_data_path = DATA_DIR_PATH + 'train_FD001.txt'
    test_data_path = DATA_DIR_PATH + 'test_FD001.txt'
    test_label_path = DATA_DIR_PATH + 'RUL_FD001.txt'
    
    stats = None
    
    def __init__(self, path=None, data_type='train', rul_ub=None, norm_type='minmax'):
        if data_type not in ['train', 'test']:
            raise ValueError("data_type must be 'train' or 'test'")
        self.path = path if path else (FD001_dataset.train_data_path if data_type == 'train' else FD001_dataset.test_data_path)
        self.is_test = (data_type == 'test')
        self.rul_ub = rul_ub
        self.norm_type = norm_type
        self.data_group = self.load_data()
        self.num_samples = len(self.data_group)
        self.targets = self.load_targets()
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        :param idx: index of the sample
        :return: a tuple (features, target) where features is a tensor of shape (sequence_length, num_features)
                 and target is a tensor of shape (sequence_length, 1)
        """
        data = self.data_group.get_group(idx + 1)
        features = data.iloc[:, 2:].values.astype(np.float32)
        target = self.targets[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    
    def load_targets(self):
        """
        Load the targets for the test dataset.
        :return: a dictionary mapping unit numbers to their corresponding RUL values
        """
        if self.is_test:
            final_rul_data = pd.read_csv(FD001_dataset.test_label_path, header=None).to_numpy().reshape(-1)
        else:
            final_rul_data = np.zeros((self.num_samples,))
        targets = {}
        for i in range(1, self.num_samples + 1):
            sequence_length = len(self.data_group.get_group(i))
            tmp = np.arange(final_rul_data[i-1] + sequence_length - 1, final_rul_data[i-1] - 1, -1).reshape(-1, 1)
            if self.rul_ub is not None:
                tmp = np.clip(tmp, 0, self.rul_ub)
            targets[i-1] = tmp
        return targets
            
    
    def load_data(self):
        data = pd.read_csv(self.path, sep='\s+', header=None, names=FD001_dataset.col_names)
        data.drop(columns=FD001_dataset.drop_labels, inplace=True)
        data_norm = self.normalize_data(data)
        data_group = data_norm.groupby(by="unit_nr")
        return data_group
        
    def normalize_data(self, df):
        """
        Normalize the input data.
        """ 
        title = df.iloc[:, 0:2]
        data = df.iloc[:, 2:]
        
        if not self.is_test:
            if self.norm_type == 'minmax':
                train_min, train_max = data.min(), data.max()
                data_norm = (data - train_min) / (train_max - train_min)
                FD001_dataset.stats = (train_min, train_max)
            elif self.norm_type == 'zscore':
                train_mean, train_std = data.mean(), data.std()
                data_norm = (data - train_mean) / train_std
                FD001_dataset.stats = (train_mean, train_std)
            else:
                raise ValueError("Normalization type must be 'minmax' or 'zscore'")
        else:
            assert FD001_dataset.stats is not None, "training data must be loaded first to get normalization statistics"
            if self.norm_type == 'minmax':
                train_min, train_max = self.stats
                data_norm = (data - train_min) / (train_max - train_min)
            elif self.norm_type == 'zscore':
                train_mean, train_std = self.stats
                data_norm = (data - train_mean) / train_std
            else:
                raise ValueError("Normalization type must be 'minmax' or 'zscore'")
        return pd.concat([title, data_norm], axis=1)
    

if __name__ == '__main__':
    train_dataset = FD001_dataset(data_type='train', rul_ub=135, norm_type='zscore')
    test_dataset = FD001_dataset(data_type='test', rul_ub=135, norm_type='zscore')

    train_dataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)