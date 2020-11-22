import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, in_chan, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(in_chan,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[1], momentum=0.9)
        self.bn2 = nn.BatchNorm2d(nStages[2], momentum=0.9)
        self.bn3 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        #self.linear = nn.Linear(nStages[3], num_classes)
        self.seg1 = nn.Conv2d(nStages[1], 128, kernel_size=3, stride=1, padding=1)
        self.seg2 = nn.Conv2d(nStages[2], 128, kernel_size=3, stride=1, padding=1)
        self.seg3 = nn.Conv2d(nStages[3], 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128*3, momentum=0.9)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.seg = nn.Conv2d(128*3, num_classes, kernel_size=3, stride=1, padding=1)
        #self.bn = nn.BatchNorm2d(nStages[3], momentum=0.9)
        #self.seg = nn.ConvTranspose2d(nStages[3], num_classes, kernel_size=8, stride=4, padding=2)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        _,_,H,W = x.size()
        out0 = self.conv1(x)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out1 = F.interpolate(self.seg1(F.relu(self.bn1(out1))), size=(H,W), mode='bilinear')
        out2 = F.interpolate(self.seg2(F.relu(self.bn2(out2))), size=(H,W), mode='bilinear')
        out3 = F.interpolate(self.seg3(F.relu(self.bn3(out3))), size=(H,W), mode='bilinear')
        #out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        #out = out.view(out.size(0), -1)
        #out = self.linear(out)
        vol = torch.cat( [out1,out2,out3], dim=1  )
        out = self.seg(self.dropout(F.relu(self.bn4(vol))))
        #out = self.seg(F.relu(self.bn(out3)))
        

        return out
