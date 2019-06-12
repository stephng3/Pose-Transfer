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
        self.norm = torch.nn.BatchNorm1d((sum(model.parts) + 2) * 256)
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

        if len(self.gpu_ids) > 0:
            mean = mean.cuda()
            std = std.cuda()

        fake_p2_norm = (fake + 1)/2 # [-1, 1] => [0, 1]
        fake_p2_norm = (fake_p2_norm - mean)/std

        input_p2_norm = (real + 1)/2 # [-1, 1] => [0, 1]
        input_p2_norm = (input_p2_norm - mean)/std

        fake_p2_norm = self.model(fake_p2_norm)[0]
        input_p2_norm = self.model(input_p2_norm)[0]

        batch = torch.cat((fake_p2_norm, input_p2_norm), 0)
        normalised_batch = self.norm(batch)
        norm_P1, norm_P2 = normalised_batch.split(normalised_batch.size()[0] // 2, 0) # split back to calc distance

        cos_loss = self.cosine_loss(norm_P1, norm_P2, torch.ones(norm_P1.size()[0]))
        loss = cos_loss * self.lambda_CL

        import pdb; pdb.set_trace()

        return loss
