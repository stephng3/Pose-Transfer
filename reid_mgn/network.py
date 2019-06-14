import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck

class MGN(nn.Module):
    def __init__(self, parts=[1,2,3], num_classes=751, input_height=384, input_width=128, **kwargs):
        super(MGN, self).__init__()

        feats = 256
        resnet = resnet50(pretrained=True)
        self.parts = parts

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        zg_p1_height = input_height // 32
        zg_p1_width = input_width // 32
        zg_p23_height = input_height // 16
        zg_p23_width = input_width // 16

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(zg_p1_height, zg_p1_width))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(zg_p23_height, zg_p23_width))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(zg_p23_height, zg_p23_width))

        zp2_height = zg_p23_height // self.parts[1]
        zp3_height = zg_p23_height // self.parts[2]

        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(zp2_height, zg_p23_width))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(zp3_height, zg_p23_width))

        _reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self._init_reduction(_reduction)
        self.reductions = nn.ModuleList([copy.deepcopy(_reduction) for i in range(sum(self.parts[1:])+3)])

        self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_1 = nn.Linear(2048, num_classes)
        self.fc_id_2048_2 = nn.Linear(2048, num_classes)

        self.fc_id_256_1 = nn.ModuleList([nn.Linear(feats, num_classes) for i in range(self.parts[1])])
        self.fc_id_256_2 = nn.ModuleList([nn.Linear(feats, num_classes) for i in range(self.parts[2])])

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        for fc in self.fc_id_256_1:
            self._init_fc(fc)
        for fc in self.fc_id_256_2:
            self._init_fc(fc)
        

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)
        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z_p2 = [zp2[:, :, i:i+1, :] for i in range(self.parts[1])]

        zp3 = self.maxpool_zp3(p3)
        z_p3 = [zp3[:, :, i:i+1, :] for i in range(self.parts[2])]

        fg_p1 = self.reductions[0](zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reductions[1](zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reductions[2](zg_p3).squeeze(dim=3).squeeze(dim=2)

        f_p2 = [r(z_p).squeeze(dim=3).squeeze(dim=2) for r,z_p in zip(self.reductions[3:], z_p2)]
        f_p3 = [r(z_p).squeeze(dim=3).squeeze(dim=2) for r,z_p in zip(self.reductions[3+self.parts[1]:], z_p3)]

        l_p1 = self.fc_id_2048_0(zg_p1.squeeze())
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze())
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze())

        ln_p2 = [l(f_p.to(next(l.parameters()).device)) for l, f_p in zip(self.fc_id_256_1,f_p2)]
        ln_p3 = [l(f_p.to(next(l.parameters()).device)) for l, f_p in zip(self.fc_id_256_2,f_p3)]

        predict = torch.cat([fg_p1, fg_p2, fg_p3] + f_p2 + f_p3, dim=1)

        retval = tuple(v for v in ([predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3] + ln_p2 + ln_p3))
        return retval
