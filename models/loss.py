# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f


class NTXentLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    @staticmethod
    def mask_correlated_samples(batch_size):
        n = 2 * batch_size
        mask = torch.ones((n, n), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_j.shape[0]
        n = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        z = f.normalize(z, dim=-1)

        mask = self.mask_correlated_samples(batch_size)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(n, 1)
        negative_samples = sim[mask].reshape(n, -1)

        labels = torch.from_numpy(np.array([0] * n)).reshape(-1).to(positive_samples.device).long()  # .float()
        logits = torch.cat((positive_samples, negative_samples), dim=1) / self.temperature

        loss = self.criterion(logits, labels)
        loss /= n
        return loss, (labels, logits)
