# -*- coding:utf-8 -*-
import os
import numpy as np
from sklearn.model_selection import KFold
from scipy.signal import butter, lfilter


def butter_bandpass_filter(signal, low_cut, high_cut, fs, order=5):
    if low_cut == 0:
        low_cut = 0.5
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    return y


def split_train_test_val_files(base_path, n_splits=5):
    # Subject Variability
    files = os.listdir(base_path)
    files = np.array(files)

    size = len(files)
    print('File Path => ' + base_path)
    print('Total Subject Size => {}'.format(size))
    kf = KFold(n_splits=n_splits)

    temp = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(files)):
        train_size = len(train_idx)
        val_point = int(train_size * 0.75)
        train_idx, val_idx = train_idx[:val_point], train_idx[val_point:]

        temp[fold] = {
            'train_paths': list([os.path.join(base_path, f_name) for f_name in files[train_idx]]),
            'ft_paths': list([os.path.join(base_path, f_name) for f_name in files[val_idx]]),
            'eval_paths': list([os.path.join(base_path, f_name) for f_name in files[test_idx]])
        }
    return temp
