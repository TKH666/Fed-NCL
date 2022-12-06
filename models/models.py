#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


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



class ResNet18Tiny(nn.Module):
    def __init__(self, args):
        super(ResNet18Tiny, self).__init__()

        tiny_model_resnet18 = models.resnet18(pretrained=False) #pretrained=TrueTrue  pretrained=False
        tiny_model_resnet18.avgpool = nn.AdaptiveAvgPool2d(1)
        tiny_model_resnet18.fc.out_features = args.num_classes

        # # Change BN to GN 
        tiny_model_resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        tiny_model_resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        tiny_model_resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        tiny_model_resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
        tiny_model_resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

        tiny_model_resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        tiny_model_resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        tiny_model_resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
        tiny_model_resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
        tiny_model_resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

        tiny_model_resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        tiny_model_resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        tiny_model_resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
        tiny_model_resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
        tiny_model_resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

        tiny_model_resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        tiny_model_resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        tiny_model_resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
        tiny_model_resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        tiny_model_resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
        self.model = tiny_model_resnet18

    def forward(self, x):
        x = self.model(x)
        return x












from utils_libs import *
import torchvision.models as models

class client_model(nn.Module):
    def __init__(self, name, args=True):
        super(client_model, self).__init__()
        self.name = name
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)
          
        if self.name == 'mnist_2NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)

        if self.name == 'mnist_lenet':
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
            self.fc3 = nn.Linear(84, self.n_cls)
            self.relu5 = nn.ReLU()
            
        if self.name == 'emnist_NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)
        
        if self.name == 'cifar10_LeNet':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)
            
        if self.name == 'cifar100_LeNet':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'tiny_imagenet_LeNet':
            self.n_cls = 200
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'tiny_imagenet_AlexNet':
            num_classes=200
            self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )


            
        if self.name == 'Resnet18':
            resnet18 = models.resnet18(pretrained=False) #pretrained=True  pretrained=True
            resnet18.fc = nn.Linear(512, 100) #10

            # # Change BN to GN 
            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

            ###################

            self.model = resnet18

        if self.name == 'tiny_Resnet18':
            tiny_model_resnet18 = models.resnet18(pretrained=False) #pretrained=TrueTrue  pretrained=False
            tiny_model_resnet18.avgpool = nn.AdaptiveAvgPool2d(1)
            tiny_model_resnet18.fc.out_features = 200

            # # Change BN to GN 
            tiny_model_resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            tiny_model_resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            tiny_model_resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            tiny_model_resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            tiny_model_resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            tiny_model_resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            tiny_model_resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            tiny_model_resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            tiny_model_resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            tiny_model_resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            tiny_model_resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            tiny_model_resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            tiny_model_resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            tiny_model_resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            tiny_model_resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            tiny_model_resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            tiny_model_resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            tiny_model_resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            tiny_model_resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            tiny_model_resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)




            self.model = tiny_model_resnet18

        if self.name == 'shakes_LSTM':
            embedding_dim = 8
            hidden_size = 100
            num_LSTM = 2
            input_length = 80
            self.n_cls = 80
            
            self.embedding = nn.Embedding(input_length, embedding_dim)
            self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
            self.fc = nn.Linear(hidden_size, self.n_cls)
              

              
        
###################################################


    def forward(self, x):
        if self.name == 'Linear':
            x = self.fc(x)
            
        if self.name == 'mnist_2NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
  
        if self.name == 'emnist_NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        if self.name == 'cifar10_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
        if self.name == 'cifar100_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
            
        if self.name == 'tiny_imagenet_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'tiny_imagenet_AlexNet':
            x = self.features(x)
            x = x.view(x.size(0), 256 * 1 * 1)
            x = self.classifier(x)

        if self.name == 'Resnet18':
            x = self.model(x)

        if self.name == 'tiny_Resnet18':
            x = self.model(x)


        if self.name == 'shakes_LSTM':
            x = self.embedding(x)
            x = x.permute(1, 0, 2) # lstm accepts in this style
            output, (h_, c_) = self.stacked_LSTM(x)
            # Choose last hidden layer
            last_hidden = output[-1,:,:]
            x = self.fc(last_hidden)

        if self.name == 'mnist_lenet':
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
