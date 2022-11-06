import torch
import torch.nn as nn
import torch.nn.functional as F

import network.resnet38d
from torch.nn import Module, Sequential, Conv2d,Softmax,Parameter

class Net(network.resnet38d.Net):
    def __init__(self, num_classes):
        super().__init__()

        self.fc8 = nn.Conv2d(4096, 2, 1, bias=False)


        self.from_scratch_layers = [self.fc8]
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

        self.query_conv = torch.nn.Conv2d(in_channels=4096, out_channels=4096 // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_channels=4096, out_channels=4096 // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1,requires_grad=True))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        N, C, H, W = x.size()
        d = super().forward_as_dict(x)
        cam = d['conv6']


        _pam = self.fc8(cam)
        m_batchsize, C, height, width = cam.size()
        proj_query = self.query_conv(cam).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(cam).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(cam).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + cam
        out = self.fc8(out)

        n,c,h,w = cam.size()
        pred = F.max_pool2d(out, kernel_size=(h, w), padding=0)
        pred = pred.view(pred.size(0), -1)

        return pred,out















    def forward_cam(self, x):
        x = super().forward(x)
        cam = self.fc8(x)

        return cam

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():



            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups