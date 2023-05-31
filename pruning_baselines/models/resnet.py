from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=54 else BasicBlock
        # ========== according to the GraSP code, we double the #filter here ============
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, n)
        self.layer2 = self._make_layer(block, 64, n, stride=2)
        self.layer3 = self._make_layer(block, 128, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(128 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    

        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x) 

        x = F.avg_pool2d(x,x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ConvBlocks(nn.Module):
    def __init__(self, input_dim = 128, hidden_dim1 = 256, hidden_dim2 = 512, output_dim = 10, is_downsample=True):
        super(ConvBlocks, self).__init__()
        #self.model = nn.Sequential(
        #        nn.Linear(input_dim, output_dim, bias = True),
        #    )

        self.is_downsample = is_downsample

        self.basic_block1_list = [
            nn.Conv2d(input_dim, hidden_dim1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block1 = nn.Sequential(*self.basic_block1_list)

        self.downsample1_list = [
            nn.Conv2d(input_dim, hidden_dim1, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ]
        self.downsample1 = nn.Sequential(*self.downsample1_list)

        self.basic_block2_list = [
            nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block2 = nn.Sequential(*self.basic_block2_list)

        self.basic_block3_list = [
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block3 = nn.Sequential(*self.basic_block3_list)

        self.downsample2_list = [
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ]
        self.downsample2 = nn.Sequential(*self.downsample2_list)

        self.basic_block4_list = [
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ]
        self.basic_block4 = nn.Sequential(*self.basic_block4_list)

        self.adaptive_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.linear = nn.Linear(in_features=hidden_dim2, out_features=output_dim, bias=True)

    def forward(self, x):
        out = self.basic_block1(x)
        if self.is_downsample:
            add_this = self.downsample1(x)
        
        out += add_this

        out = self.basic_block2(out)

        out2 = self.basic_block3(out)
        if self.is_downsample:
            add_this = self.downsample2(out)
        
        out2 += add_this

        out = self.basic_block4(out2)

        out = self.adaptive_pooling(out)
 
        d1 = out.shape[0]
        d2 = out.shape[1]
        out = self.linear(out.reshape((d1,d2)))
        return out

def pruning_baseline_resnet(**kwargs):
    return ConvBlocks(**kwargs)

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
