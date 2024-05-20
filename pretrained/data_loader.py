# -*- coding:utf-8 -*-
import mne
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


class TorchDataset(Dataset):
    def __init__(self, paths, sfreq, rfreq, scaler: bool = False):
        super().__init__()
        self.x, self.y = self.get_data(paths, sfreq, rfreq, scaler)
        self.x, self.y = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long)

    @staticmethod
    def get_data(paths, sfreq, rfreq, scaler_flag):
        info = mne.create_info(sfreq=sfreq, ch_types='eeg', ch_names=['Fp1'])
        scaler = mne.decoding.Scaler(info=info, scalings='median')
        total_x, total_y = [], []
        for path in paths:
            data = np.load(path)
            x, y = data['x'], data['y']
            x = np.expand_dims(x, axis=1)
            if scaler_flag:
                x = scaler.fit_transform(x)
            x = mne.EpochsArray(x, info=info)
            x = x.resample(rfreq)
            x = x.get_data().squeeze()
            total_x.append(x)
            total_y.append(y)
        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)
        return total_x, total_y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = torch.tensor(self.x[item])
        y = torch.tensor(self.y[item])
        return x, y
