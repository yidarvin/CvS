# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from engine.shakedrop import ShakeDrop


class ShakeBasicBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, p_shakedrop=1.0):
        super(ShakeBasicBlock, self).__init__()
        self.downsampled = stride == 2
        self.branch = self._make_branch(in_ch, out_ch, stride=stride)
        self.shortcut = not self.downsampled and None or nn.AvgPool2d(2)
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        h = self.branch(x)
        h = self.shake_drop(h)
        h0 = x if not self.downsampled else self.shortcut(x)
        pad_zero = Variable(torch.zeros(h0.size(0), h.size(1) - h0.size(1), h0.size(2), h0.size(3)).float()).cuda()
        h0 = torch.cat([h0, pad_zero], dim=1)

        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakePyramidNet(nn.Module):

    def __init__(self, in_chan=3, depth=110, alpha=270, label=10):
        super(ShakePyramidNet, self).__init__()
        in_ch = 16
        # for BasicBlock
        n_units = (depth - 2) // 6
        in_chs = [in_ch] + [in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]
        block = ShakeBasicBlock

        self.in_chs, self.u_idx = in_chs, 0
        self.ps_shakedrop = [1 - (1.0 - (0.5 / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]

        #self.c_in = nn.Conv2d(in_chan, in_chs[0], 7, stride=2, padding=3)
        self.c_in = nn.Conv2d(in_chan, in_chs[0], 3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(in_chs[0])
        self.layer1 = self._make_layer(n_units, block, 1)
        self.bn1 = nn.BatchNorm2d(self.in_chs[self.u_idx], momentum=0.9)
        self.seg1 = nn.Conv2d(self.in_chs[self.u_idx], 128, kernel_size=3, stride=1, padding=1)
        self.layer2 = self._make_layer(n_units, block, 2)
        self.bn2 = nn.BatchNorm2d(self.in_chs[self.u_idx], momentum=0.9)
        self.seg2 = nn.Conv2d(self.in_chs[self.u_idx], 128, kernel_size=3, stride=1, padding=1)
        self.layer3 = self._make_layer(n_units, block, 2)
        self.bn3 = nn.BatchNorm2d(self.in_chs[self.u_idx], momentum=0.9)
        self.seg3 = nn.Conv2d(self.in_chs[self.u_idx], 128, kernel_size=3, stride=1, padding=1)
        #self.bn_out = nn.BatchNorm2d(in_chs[-1])
        #self.fc_out = nn.Linear(in_chs[-1], label)
        #self.seg = nn.ConvTranspose2d(in_chs[-1], label, kernel_size=8, stride=4, padding=2)
        #self.linear = nn.Linear(nStages[3], num_classes)
        self.bn4 = nn.BatchNorm2d(128*3, momentum=0.9)
        self.seg = nn.Conv2d(128*3, label, kernel_size=3, stride=1, padding=1)

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        _,_,H0,W0 = x.size()
        h = self.bn_in(self.c_in(x))
        _,_,H,W = h.size()
        h1 = self.layer1(h)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        #h = F.relu(self.bn_out(h))
        #h = F.avg_pool2d(h, 8)
        #h = h.view(h.size(0), -1)
        #h = self.fc_out(h)
        #h = self.seg(h)
        out1 = F.interpolate(self.seg1(F.relu(self.bn1(h1))), size=(H,W), mode='bilinear')
        out2 = F.interpolate(self.seg2(F.relu(self.bn2(h2))), size=(H,W), mode='bilinear')
        out3 = F.interpolate(self.seg3(F.relu(self.bn3(h3))), size=(H,W), mode='bilinear')
        vol = torch.cat( [out1,out2,out3], dim=1  )
        out = self.seg(F.relu(self.bn4(vol)))
        return F.interpolate(out, size=(H0,W0), mode='bilinear')

    def _make_layer(self, n_units, block, stride=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(block(self.in_chs[self.u_idx], self.in_chs[self.u_idx+1],
                                stride, self.ps_shakedrop[self.u_idx]))
            self.u_idx, stride = self.u_idx + 1, 1
        return nn.Sequential(*layers)
