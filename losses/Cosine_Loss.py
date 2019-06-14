from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

class Cosine_Loss(nn.Module):
    def __init__(self, lambda_CL, model, gpu_ids):
        super(Cosine_Loss, self).__init__()

        self.lambda_CL = lambda_CL
        self.gpu_ids = gpu_ids

        self.model = model.eval()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, fake, real):
        if self.lambda_CL == 0:
            return torch.zeros(1).cuda(), torch.zeros(1), torch.zeros(1)

        mean = torch.FloatTensor(3)
        mean[0] = 0.485
        mean[1] = 0.456
        mean[2] = 0.406
        mean = mean.resize(1, 3, 1, 1)

        std = torch.FloatTensor(3)
        std[0] = 0.229
        std[1] = 0.224
        std[2] = 0.225
        std = std.resize(1, 3, 1, 1)

        targets = torch.ones(real.size()[0])

        if len(self.gpu_ids) > 0:
            mean = mean.cuda()
            std = std.cuda()
            targets = targets.cuda()

        fake_norm = (fake + 1)/2 # [-1, 1] => [0, 1]
        fake_norm = (fake_norm - mean)/std

        real_norm = (real + 1)/2 # [-1, 1] => [0, 1]
        real_norm = (real_norm - mean)/std

        fake_emb = self.model(fake_norm)[0]
        real_emb = self.model(real_norm)[0]

        cos_loss = self.cosine_loss(real_emb, fake_emb, targets)
        loss = cos_loss * self.lambda_CL

        return loss
