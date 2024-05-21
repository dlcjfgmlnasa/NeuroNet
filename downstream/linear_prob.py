# -*- coding:utf-8 -*-
import os
import mne
import torch
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn
from typing import List
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from models.neuronet.model import NeuroNet, NeuroNetEncoderWrapper


warnings.filterwarnings(action='ignore')


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    file_name = 'mini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_fold', default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', 'ckpt',
                                                            'ISRUC-Sleep', 'cm_eeg', file_name), type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.00005, type=float)
    return parser.parse_args()


class Classifier(nn.Module):
    def __init__(self, backbone, backbone_final_length):
        super().__init__()
        self.backbone = self.freeze_backbone(backbone)
        self.backbone_final_length = backbone_final_length
        self.feature_num = self.backbone_final_length * 2
        self.dropout_p = 0.5
        self.fc = nn.Sequential(
            nn.Linear(backbone_final_length, self.feature_num),
            nn.BatchNorm1d(self.feature_num),
            nn.ELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.feature_num, 5)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

    @staticmethod
    def freeze_backbone(backbone: nn.Module):
        for name, module in backbone.named_modules():
            for param in module.parameters():
                param.requires_grad = False
        return backbone


class Trainer(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ckpt_path = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'model', 'best_model.pth')
        self.ckpt = torch.load(self.ckpt_path, map_location='cpu')
        self.sfreq, self.rfreq = self.ckpt['hyperparameter']['sfreq'], self.ckpt['hyperparameter']['rfreq']
        self.ft_paths, self.eval_paths = self.ckpt['paths']['ft_paths'], self.ckpt['paths']['eval_paths']
        self.model = self.get_pretrained_model().to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        print('Checkpoint File Path : {}'.format(self.ckpt_path))
        train_dataset = TorchDataset(paths=self.ft_paths, sfreq=self.sfreq, rfreq=self.rfreq)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size,
                                      shuffle=True, drop_last=True)
        eval_dataset = TorchDataset(paths=self.eval_paths, sfreq=self.sfreq, rfreq=self.rfreq)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.args.batch_size, drop_last=False)

        best_model_state, best_mf1 = None, 0.0
        best_pred, best_real = [], []

        for epoch in range(self.args.epochs):
            self.model.train()
            epoch_train_loss = []
            for data in train_dataloader:
                self.optimizer.zero_grad()
                x, y = data
                x, y = x.to(device), y.to(device)

                pred = self.model(x)
                loss = self.criterion(pred, y)

                epoch_train_loss.append(float(loss.detach().cpu().item()))
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            epoch_test_loss = []
            epoch_real, epoch_pred = [], []
            for data in eval_dataloader:
                with torch.no_grad():
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    pred = pred.argmax(dim=-1)
                    real = y

                    epoch_real.extend(list(real.detach().cpu().numpy()))
                    epoch_pred.extend(list(pred.detach().cpu().numpy()))
                    epoch_test_loss.append(float(loss.detach().cpu().item()))

            epoch_train_loss, epoch_test_loss = np.mean(epoch_train_loss), np.mean(epoch_test_loss)
            eval_acc, eval_mf1 = accuracy_score(y_true=epoch_real, y_pred=epoch_pred), \
                                 f1_score(y_true=epoch_real, y_pred=epoch_pred, average='macro')

            print('[Epoch] : {0:03d} \t '
                  '[Train Loss] => {1:.4f} \t '
                  '[Evaluation Loss] => {2:.4f} \t '
                  '[Evaluation Accuracy] => {3:.4f} \t'
                  '[Evaluation Macro-F1] => {4:.4f}'.format(epoch + 1, epoch_train_loss, epoch_test_loss,
                                                            eval_acc, eval_mf1))

            if best_mf1 < eval_mf1:
                best_mf1 = eval_mf1
                best_model_state = self.model.state_dict()
                best_pred, best_real = epoch_pred, epoch_real

            self.scheduler.step()

        self.save_ckpt(best_model_state, best_pred, best_real)

    def save_ckpt(self, model_state, pred, real):
        if not os.path.exists(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'linear_prob')):
            os.makedirs(os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'linear_prob'))

        save_path = os.path.join(self.args.ckpt_path, str(self.args.n_fold), 'linear_prob', 'best_model.pth')
        torch.save({
            'backbone_name': 'NeuroNet_LinearProb',
            'model_state': model_state,
            'hyperparameter': self.args.__dict__,
            'result': {'real': real, 'pred': pred},
            'paths': {'train_paths': self.ft_paths, 'eval_paths': self.eval_paths}
        }, save_path)

    def get_pretrained_model(self):
        # 1. Prepared Pretrained Model
        model_parameter = self.ckpt['model_parameter']
        pretrained_model = NeuroNet(**model_parameter)
        pretrained_model.load_state_dict(self.ckpt['model_state'])

        # 2. Encoder Wrapper
        backbone = NeuroNetEncoderWrapper(
            fs=model_parameter['fs'], second=model_parameter['second'],
            time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
            frame_backbone=pretrained_model.frame_backbone,
            patch_embed=pretrained_model.autoencoder.patch_embed,
            encoder_block=pretrained_model.autoencoder.encoder_block,
            encoder_norm=pretrained_model.autoencoder.encoder_norm,
            cls_token=pretrained_model.autoencoder.cls_token,
            pos_embed=pretrained_model.autoencoder.pos_embed,
            final_length=pretrained_model.autoencoder.embed_dim
        )

        # 3. Generator Classifier
        model = Classifier(backbone=backbone,
                           backbone_final_length=pretrained_model.autoencoder.embed_dim)
        return model


class TorchDataset(Dataset):
    def __init__(self, paths: List, sfreq: int, rfreq: int):
        self.paths = paths
        self.info = mne.create_info(sfreq=sfreq, ch_types='eeg', ch_names=['Fp1'])
        self.xs, self.ys = self.get_data(rfreq)

    def __len__(self):
        return self.xs.shape[0]

    def get_data(self, rfreq):
        xs, ys = [], []
        for path in self.paths:
            data = np.load(path)
            x, y = data['x'], data['y']
            x = np.expand_dims(x, axis=1)
            x = mne.EpochsArray(x, info=self.info)
            x = x.resample(rfreq)
            x = x.get_data().squeeze()
            xs.append(x)
            ys.append(y)
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        return xs, ys

    def __getitem__(self, idx):
        x = torch.tensor(self.xs[idx], dtype=torch.float)
        y = torch.tensor(self.ys[idx], dtype=torch.long)
        return x, y


if __name__ == '__main__':
    augments = get_args()
    for n_fold in range(10):
        augments.n_fold = n_fold
        trainer = Trainer(augments)
        trainer.train()
