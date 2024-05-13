# -*- coding:utf-8 -*-
import os
import re
import abc
import mne
import pickle
import numpy as np
from typing import Dict


class Base(object):
    def __init__(self, labels: Dict, sfreq: float):
        self.labels = labels
        self.sfreq = sfreq

    @abc.abstractmethod
    def parser(self, path) -> (np.array, np.array):
        pass

    def save(self, scr_path: str, trg_path: str):
        x, y = self.parser(scr_path)
        np.savez(trg_path, x=x, y=y, sfreq=self.sfreq)


class SleepEDFX(Base):
    def __init__(self, labels: Dict, sfreq: float):
        super().__init__(labels=labels, sfreq=sfreq)

    def parser(self, path) -> (np.array, np.array):
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
            x, y = data['x']['Fpz-Cz EEG'], data['y']
            y = np.array([self.labels[y_] for y_ in y])
            return x, y


class SHHS(Base):
    def __init__(self, labels: Dict, sfreq: float):
        super().__init__(labels=labels, sfreq=sfreq)

    def parser(self, path) -> (np.array, np.array):
        edf_data = mne.io.read_raw_edf(path, preload=True)
        idx = edf_data.ch_names.index('EEG')
        data = edf_data.get_data()[idx]
        x = np.reshape(data, [-1, 30 * self.sfreq])

        name_ = os.path.basename(path).split('.')[0] + '-nsrr.xml'
        label_path = os.path.join(*path.split('/')[:-3], 'annotations-events-nsrr', 'shhs1', name_)
        y = self.read_annotation_regex(label_path)
        y = np.array(y)
        y = np.array([self.labels[str(y_)] for y_ in y])
        return x, y

    @staticmethod
    def read_annotation_regex(filename):
        with open(filename, 'r') as f:
            content = f.read()
        patterns_stages = re.findall(
            r'<EventType>Stages.Stages</EventType>\n' +
            r'<EventConcept>.+</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n' +
            r'<Duration>[0-9\.]+</Duration>',
            content)
        stages, starts, durations = [], [], []
        for pattern in patterns_stages:
            lines = pattern.splitlines()
            stage_line = lines[1]
            stage = int(stage_line[-16])
            start_line = lines[2]
            start = float(start_line[7:-8])
            duration_line = lines[3]
            duration = float(duration_line[10:-11])
            assert duration % 30 == 0.
            epochs_duration = int(duration) // 30
            stages += [stage]*epochs_duration
            starts += [start]
            durations += [duration]
        return stages


class ISRUC(Base):
    def __init__(self, labels: Dict, sfreq: float):
        super().__init__(labels=labels, sfreq=sfreq)

    def parser(self, path) -> (np.array, np.array):
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
            x, y = data['x'], data['y']['label_2']['stage']
            y = np.array([self.labels[y_.upper()] for y_ in y])
            for key in ['C4-A1', 'C4-M1', 'C4']:
                try:
                    x = x[key]
                    break
                except KeyError:
                    continue
            return x, y


if __name__ == '__main__':
    # 1. SleepEDFX
    src_base_path_ = os.path.join('..', '..', '..', '..', 'Dataset', 'Sleep-EDFX-2018', 'SC')
    trg_base_path_ = os.path.join('..', 'data', 'stage', 'Sleep-EDFX-2018')
    dataset = SleepEDFX(labels={'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}, sfreq=100)
    for name in os.listdir(src_base_path_):
        src_path_ = os.path.join(src_base_path_, name)
        trg_path_ = os.path.join(trg_base_path_, name.split('.')[0] + '.npz')
        dataset.save(src_path_, trg_path_)

    # # 2. SHHS
    # src_base_path_ = os.path.join('..', '..', '..', '..', 'Dataset', 'SHHS', 'polysomnography', 'edfs', 'shhs1')
    # trg_base_path_ = os.path.join('..', 'data', 'stage', 'SHHS')
    # # 0 - Wake | 1 - Stage1 | 2 - Stage2  | 3 - Stage 3/4 | 4 - Stage 3/4 | 5 - REM stage | 9 - Movement/Wake
    # dataset = SHHS(labels={'0': 0, '1': 1, '2': 2, '3': 3, '4': 3, '5': 4, '9': 0}, sfreq=125)
    # for name in open(os.path.join('.', 'selected_shhs1_files.txt')).readlines():
    #     name = name.strip() + '.edf'
    #     src_path_ = os.path.join(src_base_path_, name)
    #     trg_path_ = os.path.join(trg_base_path_, name.split('.')[0] + '.npz')
    #     dataset.save(src_path_, trg_path_)

    # # 3. ISRUC
    # src_base_path_ = os.path.join('..', '..', '..', '..', 'Dataset', 'ISRUC-Sleep', 'Group1')
    # trg_base_path_ = os.path.join('..', 'data', 'stage', 'ISRUC-Sleep')
    # dataset = ISRUC(labels={'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}, sfreq=200)
    # for name in os.listdir(src_base_path_):
    #     src_path_ = os.path.join(src_base_path_, name)
    #     trg_path_ = os.path.join(trg_base_path_, name.split('.')[0] + '.npz')
    #     dataset.save(src_path_, trg_path_)

