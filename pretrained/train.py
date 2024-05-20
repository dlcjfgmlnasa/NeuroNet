# -*- coding:utf-8 -*-
import os
import mne
import torch
import random
import shutil
import argparse
import warnings
import numpy as np
import torch.optim as opt
from models.utils import model_size
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from dataset.utils import split_train_test_val_files
from pretrained.data_loader import TorchDataset
from models.neuronet.model import NeuroNet


warnings.filterwarnings(action='ignore')


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', 'data', 'stage', 'Sleep-EDFX-2018'))
    parser.add_argument('--k_splits', default=5)
    parser.add_argument('--n_fold', default=0, choices=[0, 1, 2, 3, 4])

    # Dataset Hyperparameter
    parser.add_argument('--sfreq', default=100, type=int)
    parser.add_argument('--rfreq', default=100, type=int)
    parser.add_argument('--data_scaler', default=False, type=bool)

    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=30, type=int)
    parser.add_argument('--train_warmup_epoch', type=int, default=100)
    parser.add_argument('--train_base_learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--train_batch_accumulation', default=1, type=int)

    # Model Hyperparameter
    parser.add_argument('--second', default=30, type=int)
    parser.add_argument('--time_window', default=3, type=int)
    parser.add_argument('--time_step', default=0.375, type=int)

    #  >> 1. NeuroNet-M Hyperparameter
    # parser.add_argument('--encoder_dim', default=512, type=int)
    # parser.add_argument('--encoder_heads', default=8, type=int)
    # parser.add_argument('--encoder_depths', default=4, type=int)
    # parser.add_argument('--decoder_embed_dim', default=192, type=int)
    # parser.add_argument('--decoder_heads', default=8, type=int)
    # parser.add_argument('--decoder_depths', default=1, type=int)

    #  >> 2. NeuroNet-B Hyperparameter
    parser.add_argument('--encoder_embed_dim', default=768, type=int)
    parser.add_argument('--encoder_heads', default=8, type=int)
    parser.add_argument('--encoder_depths', default=4, type=int)
    parser.add_argument('--decoder_embed_dim', default=256, type=int)
    parser.add_argument('--decoder_heads', default=8, type=int)
    parser.add_argument('--decoder_depths', default=3, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)

    parser.add_argument('--projection_hidden', default=[1024, 512], type=list)
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--mask_ratio', default=0.8, type=float)
    parser.add_argument('--print_point', default=20, type=int)
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', 'ckpt', 'Sleep-EDFX'), type=str)
    parser.add_argument('--model_name', default='mini')
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = NeuroNet(
            fs=args.rfreq, second=args.second, time_window=args.time_window, time_step=args.time_step,
            encoder_embed_dim=args.encoder_embed_dim, encoder_heads=args.encoder_heads, encoder_depths=args.encoder_depths,
            decoder_embed_dim=args.decoder_embed_dim, decoder_heads=args.decoder_heads,
            decoder_depths=args.decoder_depths, projection_hidden=args.projection_hidden, temperature=args.temperature
        ).to(device)
        print('Model Size : {0:.2f}MB'.format(model_size(self.model)))

        self.eff_batch_size = self.args.train_batch_size * self.args.train_batch_accumulation
        self.lr = self.args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.train_paths, self.val_paths, self.eval_paths = self.data_paths()
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.train_epochs)
        self.tensorboard_path = os.path.join(self.args.ckpt_path, self.args.model_name,
                                             str(self.args.n_fold), 'tensorboard')

        # remote tensorboard files
        if os.path.exists(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)

        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)

        print('Frame Size : {}'.format(self.model.num_patches))
        print('Leaning Rate : {0}'.format(self.lr))
        print('Validation Paths : {0}'.format(len(self.val_paths)))
        print('Evaluation Paths : {0}'.format(len(self.eval_paths)))

    def train(self):
        print('K-Fold : {}/{}'.format(self.args.n_fold + 1, self.args.k_splits))
        train_dataset = TorchDataset(paths=self.train_paths, sfreq=self.args.sfreq, rfreq=self.args.rfreq,
                                     scaler=self.args.data_scaler)
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True)
        val_dataset = TorchDataset(paths=self.val_paths, sfreq=self.args.sfreq, rfreq=self.args.rfreq,
                                   scaler=self.args.data_scaler)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.train_batch_size, drop_last=True)
        eval_dataset = TorchDataset(paths=self.eval_paths, sfreq=self.args.sfreq, rfreq=self.args.rfreq,
                                    scaler=self.args.data_scaler)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.args.train_batch_size, drop_last=True)

        total_step = 0
        best_model_state, best_score = self.model.state_dict(), 0

        for epoch in range(self.args.train_epochs):
            step = 0
            self.model.train()
            self.optimizer.zero_grad()

            for x, _ in train_dataloader:
                x = x.to(device)
                out = self.model(x, mask_ratio=self.args.mask_ratio)
                recon_loss, contrastive_loss, (cl_labels, cl_logits) = out

                loss = recon_loss + self.args.alpha * contrastive_loss
                loss.backward()

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (total_step + 1) % self.args.print_point == 0:
                    print('[Epoch] : {0:03d}  [Step] : {1:06d}  '
                          '[Reconstruction Loss] : {2:02.4f}  [Contrastive Loss] : {3:02.4f}  '
                          '[Total Loss] : {4:02.4f}  [Contrastive Acc] : {5:02.4f}'.format(
                            epoch, total_step + 1, recon_loss, contrastive_loss, loss,
                            self.compute_metrics(cl_logits, cl_labels)))

                self.tensorboard_writer.add_scalar('Reconstruction Loss', recon_loss, total_step)
                self.tensorboard_writer.add_scalar('Contrastive loss', contrastive_loss, total_step)
                self.tensorboard_writer.add_scalar('Total loss', loss, total_step)

                step += 1
                total_step += 1

            val_acc, val_mf1 = self.linear_probing(val_dataloader, eval_dataloader)

            if val_mf1 > best_score:
                best_model_state = self.model.state_dict()
                best_score = val_mf1

            print('[Epoch] : {0:03d} \t [Accuracy] : {1:2.4f} \t [Macro-F1] : {2:2.4f} \n'.format(
                epoch, val_acc * 100, val_mf1 * 100))
            self.tensorboard_writer.add_scalar('Validation Accuracy', val_acc, total_step)
            self.tensorboard_writer.add_scalar('Validation Macro-F1', val_mf1, total_step)

            self.optimizer.step()
            self.scheduler.step()

        self.save_ckpt(model_state=best_model_state)

    def linear_probing(self, val_dataloader, eval_dataloader):
        self.model.eval()
        (train_x, train_y), (test_x, test_y) = self.get_latent_vector(val_dataloader), \
                                               self.get_latent_vector(eval_dataloader)
        pca = PCA(n_components=50)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)

        model = KNeighborsClassifier()
        model.fit(train_x, train_y)

        out = model.predict(test_x)
        acc, mf1 = accuracy_score(test_y, out), f1_score(test_y, out, average='macro')
        self.model.train()
        return acc, mf1

    def get_latent_vector(self, dataloader):
        total_x, total_y = [], []
        with torch.no_grad():
            for data in dataloader:
                x, y = data
                x, y = x.to(device), y.to(device)
                latent = self.model.forward_latent(x)
                total_x.append(latent.detach().cpu().numpy())
                total_y.append(y.detach().cpu().numpy())
        total_x, total_y = np.concatenate(total_x, axis=0), np.concatenate(total_y, axis=0)
        return total_x, total_y

    def save_ckpt(self, model_state):
        ckpt_path = os.path.join(self.args.ckpt_path, self.args.model_name, str(self.args.n_fold), 'model')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        torch.save({
            'model_name': 'NeuroNet',
            'model_state': model_state,
            'model_parameter': {
                'fs': self.args.rfreq, 'second': self.args.second,
                'time_window': self.args.time_window, 'time_step': self.args.time_step,
                'encoder_embed_dim': self.args.encoder_embed_dim, 'encoder_heads': self.args.encoder_heads,
                'encoder_depths': self.args.encoder_depths,
                'decoder_embed_dim': self.args.decoder_embed_dim, 'decoder_heads': self.args.decoder_heads,
                'decoder_depths': self.args.decoder_depths,
                'projection_hidden': self.args.projection_hidden, 'temperature': self.args.temperature
            },
            'hyperparameter': self.args.__dict__,
            'paths': {'train_paths': self.train_paths, 'ft_paths': self.val_paths, 'eval_paths': self.eval_paths}
        }, os.path.join(ckpt_path, 'best_model.pth'))

    def data_paths(self):
        kf = split_train_test_val_files(base_path=self.args.base_path, n_splits=self.args.k_splits)

        paths = kf[self.args.n_fold]
        train_paths, ft_paths, eval_paths = paths['train_paths'], paths['ft_paths'], paths['eval_paths']
        return train_paths, ft_paths, eval_paths

    @staticmethod
    def compute_metrics(output, target):
        output = output.argmax(dim=-1)
        accuracy = torch.mean(torch.eq(target, output).to(torch.float32))
        return accuracy


if __name__ == '__main__':
    augments = get_args()
    for n_fold in range(augments.k_splits):
        augments.n_fold = n_fold
        trainer = Trainer(augments)
        trainer.train()
