import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from helperFunctionsJoint import seed_torch

import os
import argparse

from cifar100_models.resnet import resnet18


if __name__ == '__main__':
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print("device : ", device)

    seed_torch(2)

    transform_normal = transforms.Compose([

        transforms.CenterCrop(24),
        torchvision.transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.4865, 0.4409), (0.267, 0.2564, 0.2761)),
    ])

    transform_rotate = transforms.Compose([

        transforms.RandomRotation((30,30.00001)),
        transforms.CenterCrop(24),
        torchvision.transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.4865, 0.4409), (0.267, 0.2564, 0.2761)),
    ])

    trainset_normal = torchvision.datasets.CIFAR100(
        root='./data_cifar', train=True, download=True, transform=transform_normal)
    trainset_rotate = torchvision.datasets.CIFAR100(
        root='./data_cifar', train=True, download=True, transform=transform_rotate)

    trainloader_normal = torch.utils.data.DataLoader(
        trainset_normal, batch_size=100, shuffle=True, num_workers=2)
    trainloader_rotate = torch.utils.data.DataLoader(
        trainset_rotate, batch_size=100, shuffle=True, num_workers=2)

    # print(trainset_normal.data.shape)
    # data = trainset_normal.data / 255 # data is numpy array
    # mean = data.mean(axis = (0,1,2)) 
    # std = data.std(axis = (0,1,2))
    # print(f"Mean : {mean}   STD: {std}")
    # # print(trainset_normal.targets)
    # #Mean : [0.50707516 0.48654887 0.44091784]   STD: [0.26733429 0.25643846 0.27615047]


    testset_normal = torchvision.datasets.CIFAR100(
        root='./data_cifar', train=False, download=True, transform=transform_normal)
    testset_rotate = torchvision.datasets.CIFAR100(
        root='./data_cifar', train=False, download=True, transform=transform_rotate)
        
    testloader_normal = torch.utils.data.DataLoader(
        testset_normal, batch_size=100, shuffle=True, num_workers=2)
    testloader_rotate = torch.utils.data.DataLoader(
        testset_rotate, batch_size=100, shuffle=True, num_workers=2)


    classes = ('unrotated','rotated')


    net = resnet18(pretrained=True)
    # net = models.resnet18(pretrained=True)

    for param in net.parameters():
        param.requires_grad = False

    net = net.to(device)
    hidden_size = 128
    dim2 = 8
    n_components = 8192

    feature_extractor = torch.nn.Sequential(*list(net.children())[:-4])

    d = {}


    train_features = np.zeros((50000,hidden_size,dim2,dim2))
    train_labels = np.zeros((50000))

    test_features = np.zeros((10000,hidden_size,dim2,dim2))
    test_labels = np.zeros((10000))

    for batch_idx, (inputs, targets) in enumerate(testloader_normal):
        if(batch_idx >= 50): break
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        test_features[batch_idx*200:batch_idx*200+100] = features.cpu().numpy()
        test_labels[batch_idx*200:batch_idx*200+100] = np.zeros(100)
    for batch_idx, (inputs, targets) in enumerate(testloader_rotate):
        if(batch_idx >= 50): break
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        test_features[batch_idx*200+100:batch_idx*200+200] = features.cpu().numpy()
        test_labels[batch_idx*200+100:batch_idx*200+200] = np.ones(100)

    test_features,test_labels = shuffle(test_features,test_labels,random_state=0)

    d['resnet18_test_features'] = test_features
    d['test_labels'] = test_labels


    for batch_idx, (inputs, targets) in enumerate(trainloader_normal):
        if (batch_idx >= 125): break
        #if(batch_idx >= 250): break
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        train_features[batch_idx*200:batch_idx*200+100] = features.cpu().numpy()
        train_labels[batch_idx*200:batch_idx*200+100] = np.zeros(100)
    for batch_idx, (inputs, targets) in enumerate(trainloader_rotate):
        if (batch_idx >= 125): break
        #if(batch_idx >= 250): break
        inputs = inputs.to(device)
        features = feature_extractor(inputs).squeeze()
        train_features[batch_idx*200+100:batch_idx*200+200] = features.cpu().numpy()
        train_labels[batch_idx*200+100:batch_idx*200+200] = np.ones(100)

    train_features,train_labels = shuffle(train_features,train_labels,random_state=0)

    d['resnet18_train_features'] = train_features
    d['train_labels'] = train_labels


    for k,v in d.items():
        print(k,v.shape)

    torch.save(d,'cifar100-rotate_without_pca_l4.pth')