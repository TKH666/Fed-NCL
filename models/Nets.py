#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models



class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
class CNN_6(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        #input channel cifer is 3, mnist is 1
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN_6, self).__init__()
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(self.bn1(h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(self.bn2(h), negative_slope=0.01)
        h = self.c3(h)
        h = F.leaky_relu(self.bn3(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(self.bn4(h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(self.bn5(h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(self.bn6(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(self.bn7(h), negative_slope=0.01)
        h = self.c8(h)
        h = F.leaky_relu(self.bn8(h), negative_slope=0.01)
        h = self.c9(h)
        h = F.leaky_relu(self.bn9(h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        feature = h.view(h.size(0), h.size(1))

        logit = self.l_c1(feature)

        if self.top_bn:
            logit = call_bn(self.bn_c1, logit)

        return logit

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LenetMnist(nn.Module):
    def __init__(self, args):
        super(LenetMnist, self).__init__()

        self.n_cls = 10
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, args.num_classes)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.relu5(x)
        return x


class ResNet18Cifar10(nn.Module):
    def __init__(self, args):
        super(ResNet18Cifar10, self).__init__()

        resnet18 = models.resnet18(pretrained=False) #pretrained=True  pretrained=True
        resnet18.fc = nn.Linear(512, args.num_classes) #10

        # # Change BN to GN 
        # resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        # resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        # resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        # resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        # resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        # resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

        # resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

        # resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

        ###################

        self.model = resnet18

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet18Cifar100(nn.Module):
    def __init__(self, args):
        super(ResNet18Cifar100, self).__init__()

        resnet18 = models.resnet18(pretrained=False) #pretrained=True  pretrained=True
        resnet18.fc = nn.Linear(512, args.num_classes) #10

        # # Change BN to GN 
        # resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        # resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        # resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        # resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        # resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        # resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

        # resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

        # resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

        ###################

        self.model = resnet18

    def forward(self, x):
        x = self.model(x)
        return x


class VGG16Cifar10(nn.Module):
    def __init__(self, args):
        super(VGG16Cifar10, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #2
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #4
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #5
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #6
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #7
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #8
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #9
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #10
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #11
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #12
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #13
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),
            )
        self.classifier = nn.Sequential(
            #14
            nn.Linear(512,4096),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #16
            nn.Linear(4096,args.num_classes),
            )
        #self.classifier = nn.Linear(512, 10)
 
    def forward(self, x):
        out = self.features(x) 
#        print(out.shape)
        out = out.view(out.size(0), -1)
#        print(out.shape)
        out = self.classifier(out)
#        print(out.shape)
        return out


class ResNet18Tiny(nn.Module):
    def __init__(self, args):
        super(ResNet18Tiny, self).__init__()

        tiny_model_resnet18 = models.resnet18(pretrained=False) #pretrained=TrueTrue  pretrained=False
        tiny_model_resnet18.avgpool = nn.AdaptiveAvgPool2d(1)
        tiny_model_resnet18.fc.out_features = args.num_classes

        # # Change BN to GN 
        # tiny_model_resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        # tiny_model_resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        # tiny_model_resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        # tiny_model_resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        # tiny_model_resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        # tiny_model_resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # tiny_model_resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # tiny_model_resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # tiny_model_resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        # tiny_model_resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

        # tiny_model_resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # tiny_model_resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # tiny_model_resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # tiny_model_resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        # tiny_model_resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

        # tiny_model_resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # tiny_model_resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # tiny_model_resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # tiny_model_resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        # tiny_model_resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        self.model = tiny_model_resnet18

    def forward(self, x):
        x = self.model(x)
        return x


