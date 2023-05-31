import os
import PIL
import torch
import pickle
import random
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class CIFAR10_ROTATE(Dataset):
    def __init__(self, train):
        super(CIFAR10_ROTATE, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 128 #128,8,8
        self.num_features = 128 #128,8,8
        self.num_classes = 2
        if self.train:
            self.data_size = 25000
            # cifar = torch.load('../18_cifar/PyTorch_CIFAR10/cifar-rotate_without_pca_l4_half.pth')
            cifar = torch.load('./PyTorch_CIFAR10-ROTATE/cifar-rotate_without_pca_l4_10C_half_2.pth')
            # self.data = torch.Tensor(cifar['final_train_features'])
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 10000
            # cifar = torch.load('../18_cifar/PyTorch_CIFAR10/cifar-rotate_without_pca_l4_half.pth')
            cifar = torch.load('./PyTorch_CIFAR10-ROTATE/cifar-rotate_without_pca_l4_10C_half_2.pth')
            # self.data = torch.Tensor(cifar['final_test_features'])
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

    # def nearest(self,x):
    #     idx = torch.searchsorted(self.sortedX, x)
    #     return self.indices[idx]

    def plot(self):
        pass
        # plt.show()
        # plt.savefig('data3.png')


class CIFAR100_ROTATE(Dataset):
    def __init__(self, train):
        super(CIFAR100_ROTATE, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 128 #128,8,8
        self.num_features = 128 #128,8,8
        self.num_classes = 2
        if self.train:
            self.data_size = 50000
            # cifar = torch.load('../18_cifar/PyTorch_CIFAR10/cifar-rotate_without_pca_l4_half.pth')
            cifar = torch.load('./PyTorch_CIFAR100-ROTATE/cifar100-rotate_without_pca_l4.pth')
            # self.data = torch.Tensor(cifar['final_train_features'])
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 10000
            # cifar = torch.load('../18_cifar/PyTorch_CIFAR10/cifar-rotate_without_pca_l4_half.pth')
            cifar = torch.load('./PyTorch_CIFAR100-ROTATE/cifar100-rotate_without_pca_l4.pth')
            # self.data = torch.Tensor(cifar['final_test_features'])
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

    # def nearest(self,x):
    #     idx = torch.searchsorted(self.sortedX, x)
    #     return self.indices[idx]

    def plot(self):
        pass
        # plt.show()
        # plt.savefig('data3.png')


class STL10_ROTATE(Dataset):
    def __init__(self, train):
        super(STL10_ROTATE, self).__init__()
        if train:
            seed_torch(0)
        else:
            seed_torch(1)
        self.train = train
        self.num_attributes = 128 #128,8,8
        self.num_features = 128 #128,8,8
        self.num_classes = 2
        if self.train:
            self.data_size = 16000*2
            cifar = torch.load('./PyTorch_STL10-ROTATE/stl_l4_32_10K_half.pth')
            self.data = torch.Tensor(cifar['resnet18_train_features'])
            self.labels = torch.Tensor(cifar['train_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        else:
            self.data_size = 10000
            cifar = torch.load('./PyTorch_STL10-ROTATE/stl_l4_32_10K_half.pth')            
            self.data = torch.Tensor(cifar['resnet18_test_features'])
            self.labels = torch.Tensor(cifar['test_labels'])
            self.labels = self.labels.type(torch.LongTensor)
        
    def __getitem__(self, i):
        return i, self.data[i], self.labels[i]
        
    def __len__(self):
        return self.data.shape[0]

    # def nearest(self,x):
    #     idx = torch.searchsorted(self.sortedX, x)
    #     return self.indices[idx]

    def plot(self):
        pass
        # plt.show()
        # plt.savefig('data3.png')