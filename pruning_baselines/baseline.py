from __future__ import print_function

import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100/TinyImagenet Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='results/', type=str)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# ========== Sanity Check: randLabel & shufflePixel on CIFAR-10 ============
# ========== --randLabel will make the  ============
# ========== To reproduce the experiment in our paper, you just need to specify --shufflePixel 1 ============
# ========== See code between line 111-167 ============
parser.add_argument('--randLabel',type=int, default=0,help = 'Using randLabel Dataset for LT training')
parser.add_argument('--shufflePixel',type=int, default=0,help = 'Using shufflePixel AND RANDLABEL Dataset for LT training')
# ========== Ablation Study: Half Dataset on CIFAR-10 ============
# ========== can specify the --max_batch_idx argument to nonzero. If so, the SHUFFLE attribute of the trainloader ============
# ========== will be CLOSED and the training procedure will only use the 0~max_batch_idx-th-batch-traindata ============
# ========== in our experiments we set this number to 390 (since totally exists 50,000/64 ~ 781 full mini-batches) ============
parser.add_argument('--max_batch_idx',type = int, default = 0,help = 'Control the training data size')

parser.add_argument('--presto_baseline', action='store_true')

# ConvBlocks dimensions
parser.add_argument('--hidden_dim_1', type = int, default = 512)
parser.add_argument('--hidden_dim_2', type = int, default = 1024)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.presto_baseline or args.dataset == 'cifar10' or args.dataset == 'cifar100' or \
args.dataset == 'pathmnist' or args.dataset== 'svhn' or\
args.dataset == 'tinyimagenet', 'Dataset can only be cifar10 or cifar100 or tinyimagenet.'


gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
use_cuda = torch.cuda.is_available()

max_gpu_usage = torch.cuda.memory_allocated('cuda:0')

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy




# import numpy as np


# from torch.utils.data.dataset import Dataset
# class CIFAR10(Dataset):
#     def __init__(self, train):
#         super(CIFAR10, self).__init__()
#         # if train:      ##They seed using their function elsewhere (Verify)
#         #     seed_torch(0)
#         # else:
#         #     seed_torch(1)
#         self.train = train
#         self.num_attributes = 256 #256,4,4
#         self.num_features = 256 # 256,4,4
#         self.num_classes = 10
#         if self.train:
#             self.data_size = 50000
#             cifar = torch.load('../PyTorch_CIFAR10/cifar_without_pca_l4.pth')
#             # self.data = torch.Tensor(cifar['final_train_features'])
#             self.data = torch.Tensor(cifar['resnet18_train_features'])
#             self.labels = torch.Tensor(cifar['train_labels'])
#             self.labels = self.labels.type(torch.LongTensor)
#         else:
#             self.data_size = 10000
#             cifar = torch.load('../PyTorch_CIFAR10/cifar_without_pca_l4.pth')
#             # self.data = torch.Tensor(cifar['final_test_features'])
#             self.data = torch.Tensor(cifar['resnet18_test_features'])
#             self.labels = torch.Tensor(cifar['test_labels'])
#             self.labels = self.labels.type(torch.LongTensor)

#         # print('Initialised Baseline Dataset')
#         # print(self.data.shape, self.labels.shape)
#         # print(self.data[0].shape, self.labels[0])
        
#     def __getitem__(self, i):
#         return self.data[i], self.labels[i] #Changed from i,d[i],l[i] to match Pytorch Cifar10
        
#     def __len__(self):
#         return self.data.shape[0]

# class BloodMNIST(Dataset):
#     def __init__(self, train):
#         super(BloodMNIST, self).__init__()
#         if train:
#             seed_torch(0)
#         else:
#             seed_torch(1)
#         self.train = train
#         self.num_attributes = 128 # 128,4,4
#         self.num_features = 128 # 128,4,4
#         self.num_classes = 8
#         if self.train:
#             self.data_size = 11959
#             cifar = torch.load('../../21_bloodmnist/PyTorch_BloodMNIST/bloodmnist_without_pca_l4.pth')
#             # self.data = torch.Tensor(cifar['final_train_features'])
#             self.data = torch.Tensor(cifar['resnet18_train_features'])
#             self.labels = torch.Tensor(cifar['train_labels'])
#             self.labels = self.labels.type(torch.LongTensor)
#         else:
#             self.data_size = 3421
#             cifar = torch.load('../../21_bloodmnist/PyTorch_BloodMNIST/bloodmnist_without_pca_l4.pth')
#             # self.data = torch.Tensor(cifar['final_test_features'])
#             self.data = torch.Tensor(cifar['resnet18_test_features'])
#             self.labels = torch.Tensor(cifar['test_labels'])
#             self.labels = self.labels.type(torch.LongTensor)
        
#     def __getitem__(self, i):
#         return self.data[i], self.labels[i]
        
#     def __len__(self):
#         return self.data.shape[0]



# class PathMNIST(Dataset):
#     def __init__(self, train):
#         super(PathMNIST, self).__init__()
#         if train:
#             seed_torch(0)
#         else:
#             seed_torch(1)
#         self.train = train
#         self.num_attributes = 64 # 64,7,7
#         self.num_features = 64 # 64,7,7
#         self.num_classes = 9
#         if self.train:
#             self.data_size = 89996
#             cifar = torch.load('../../21_pathmnist/PyTorch_PathMNIST/pathmnist_without_pca_l5.pth')
#             # self.data = torch.Tensor(cifar['final_train_features'])
#             self.data = torch.Tensor(cifar['resnet18_train_features'])
#             self.labels = torch.Tensor(cifar['train_labels'])
#             self.labels = self.labels.type(torch.LongTensor)
#         else:
#             self.data_size = 7180
#             cifar = torch.load('../../21_pathmnist/PyTorch_PathMNIST/pathmnist_without_pca_l5.pth')
#             # self.data = torch.Tensor(cifar['final_test_features'])
#             self.data = torch.Tensor(cifar['resnet18_test_features'])
#             self.labels = torch.Tensor(cifar['test_labels'])
#             self.labels = self.labels.type(torch.LongTensor)
        
#     def __getitem__(self, i):
#         return self.data[i], self.labels[i]
        
#     def __len__(self):
#         return self.data.shape[0]

# ####

# class ConvBlocks(nn.Module):
#     def __init__(self, input_dim = 128, hidden_dim1 = 256, hidden_dim2 = 512, output_dim = 10, is_downsample=True):
#         super(ConvBlocks, self).__init__()
#         #self.model = nn.Sequential(
#         #        nn.Linear(input_dim, output_dim, bias = True),
#         #    )

#         self.is_downsample = is_downsample

#         self.basic_block1_list = [
#             nn.Conv2d(input_dim, hidden_dim1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
#             nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#             nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#         ]
#         self.basic_block1 = nn.Sequential(*self.basic_block1_list)

#         self.downsample1_list = [
#             nn.Conv2d(input_dim, hidden_dim1, kernel_size=(1, 1), stride=(2, 2), bias=False),
#             nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         ]
#         self.downsample1 = nn.Sequential(*self.downsample1_list)

#         self.basic_block2_list = [
#             nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#             nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#             nn.BatchNorm2d(hidden_dim1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#         ]
#         self.basic_block2 = nn.Sequential(*self.basic_block2_list)

#         self.basic_block3_list = [
#             nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
#             nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#             nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#         ]
#         self.basic_block3 = nn.Sequential(*self.basic_block3_list)

#         self.downsample2_list = [
#             nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=(1, 1), stride=(2, 2), bias=False),
#             nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         ]
#         self.downsample2 = nn.Sequential(*self.downsample2_list)

#         self.basic_block4_list = [
#             nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#             nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
#             nn.BatchNorm2d(hidden_dim2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#         ]
#         self.basic_block4 = nn.Sequential(*self.basic_block4_list)

#         self.adaptive_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

#         self.linear = nn.Linear(in_features=hidden_dim2, out_features=output_dim, bias=True)

#     def forward(self, x):
#         out = self.basic_block1(x)
#         if self.is_downsample:
#             add_this = self.downsample1(x)
        
#         out += add_this

#         out = self.basic_block2(out)

#         out2 = self.basic_block3(out)
#         if self.is_downsample:
#             add_this = self.downsample2(out)
        
#         out2 += add_this

#         out = self.basic_block4(out2)

#         out = self.adaptive_pooling(out)
 
#         d1 = out.shape[0]
#         d2 = out.shape[1]
#         out = self.linear(out.reshape((d1,d2)))
#         return out

# from presto import SVHN, ConvBlocks, DermaMNIST
from myDatasets import *
from myModels import *


class CIFAR10RandomLabels(datasets.CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
      Default 1.0. The probability of a label being replaced with
      random label.
    num_classes: int
      Default 10. The number of classes in the dataset.
    """

    def __init__(self, corrupt_prob=1.0, num_classes=10, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        # ========== Random Label Operation ============
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        targets = [int(x) for x in labels]
        self.targets = targets
        
        # ========== Random (Shuffle) Pixel Operation ============
        if args.shufflePixel != 0:
            print('********************* DEBUG PRINT : ADDITION : SHUFFLE PIXEL ************************')
            xs = torch.tensor(self.data)
            Size = xs.size()
            # e.g. for CIFAR10, is 50000 * 32 * 32 * 3
            xs = xs.reshape(Size[0],-1)
            for i in range(Size[0]):
                xs[i] = xs[i][torch.randperm(xs[i].nelement())]
            xs = xs.reshape(Size)
            xs = xs.numpy()
            self.data = xs

class CIFAR100RandomLabels(datasets.CIFAR100):
    """CIFAR100 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
      Default 1.0. The probability of a label being replaced with
      random label.
    num_classes: int
      Default 100. The number of classes in the dataset.
    """

    def __init__(self, corrupt_prob=1.0, num_classes=100, **kwargs):
        super(CIFAR100RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        targets = [int(x) for x in labels]
        self.targets = targets
        
        if args.shufflePixel != 0:
            print('********************* DEBUG PRINT : ADDITION : SHUFFLE PIXEL ************************')
            xs = torch.tensor(self.data)
            Size = xs.size()
            # e.g. for CIFAR100, is 50000 * 32 * 32 * 3
            xs = xs.reshape(Size[0],-1)
            for i in range(Size[0]):
                xs[i] = xs[i][torch.randperm(xs[i].nelement())]
            xs = xs.reshape(Size)
            xs = xs.numpy()
            self.data = xs

def main():
    global max_gpu_usage
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    os.makedirs(args.save_dir, exist_ok=True)

    if args.presto_baseline:

        if args.dataset == 'cifar10-rotate':
            print('==> Preparing dataset %s' % 'CIFAR10 Rotated Presto Baseline')
            ##Features in our dataset are already transformed
            num_classes = 2
            trainset = CIFAR10_ROTATE(train = True)
            trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle = True,  num_workers=args.workers)
            testset = CIFAR10_ROTATE(train = False)
            testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle = False,  num_workers=args.workers)

        elif args.dataset == 'cifar100-rotate':
            print('==> Preparing dataset %s' % 'CIFAR100 Rotated Presto Baseline')
            ##Features in our dataset are already transformed
            num_classes = 2
            trainset = CIFAR100_ROTATE(train = True)
            trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle = True,  num_workers=args.workers)
            testset = CIFAR100_ROTATE(train = False)
            testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle = False,  num_workers=args.workers)

        elif args.dataset == 'stl-rotate':
            print('==> Preparing dataset %s' % 'STL10 Rotated Presto Baseline')
            ##Features in our dataset are already transformed
            num_classes = 2
            trainset = STL10_ROTATE(train = True)
            trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle = True,  num_workers=args.workers)
            testset = STL10_ROTATE(train = False)
            testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle = False,  num_workers=args.workers)

    # Data
    else:
        
        print('==> Preparing dataset %s' % args.dataset)
        if args.dataset == 'cifar10':
            transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

            transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
            dataloader = datasets.CIFAR10
            num_classes = 10
        elif args.dataset == 'cifar100':
            dataloader = datasets.CIFAR100
            num_classes = 100
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        elif args.dataset == 'tinyimagenet':
            args.schedule = [150,225]
            num_classes = 200
            tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
            tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(tiny_mean, tiny_std)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(tiny_mean, tiny_std)])
            args.workers = 16
            args.epochs = 300
            
            
        if args.randLabel != 0 or args.shufflePixel != 0:
            assert args.dataset == 'cifar10' or args.dataset == 'cifar100','randLabel/shufflePixel can only be used together with cifar10/100.'
            print('###################### DEBUG PRINT : USING RANDLABEL TRAINING ####################')
            if args.dataset == 'cifar10':
                trainset = CIFAR10RandomLabels(root='./data', train=True, download=True, transform=transform_train)
            else:
                trainset = CIFAR100RandomLabels(root='./data', train=True, download=True, transform=transform_train)
        elif args.dataset != 'tinyimagenet':
            trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        else:
            trainset = datasets.ImageFolder('./data' + '/tiny_imagenet/train', transform=transform_train)
            
        if args.max_batch_idx == 0:
            trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        else:
            trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)
        
        if args.dataset != 'tinyimagenet':
            testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
            testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        else:
            testset = datasets.ImageFolder('./data' + '/tiny_imagenet/val', transform=transform_test)
            testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False,
                                                num_workers=args.workers)
        
        
        

    # Model
    if args.presto_baseline:
        print("==> Creating model '{}'".format('Resnet18 Presto Skyline'))

        if args.dataset == 'cifar10' or 'rotate' in args.dataset:
            model = ConvBlocks_CIFAR10(input_dim=trainset.num_features, hidden_dim1=args.hidden_dim_1, hidden_dim2=args.hidden_dim_2,\
            output_dim=trainset.num_classes, is_downsample=True)
            # model = ConvBlocks_CIFAR10(input_dim=trainset.num_features, hidden_dim1=256, hidden_dim2=512,\
            # output_dim=trainset.num_classes, is_downsample=True)
        elif args.dataset != 'cifar10':
            model = ConvBlocks(input_dim=trainset.num_features, hidden_dim1=256, hidden_dim2=512, hidden_dim3=1024,\
            output_dim=trainset.num_classes, is_downsample=True)
            # model = ConvBlocks(input_dim=trainset.num_features, hidden_dim1=128, hidden_dim2=256, hidden_dim3=512,\
            # output_dim=trainset.num_classes, is_downsample=True)

        model.cuda()
        max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated('cuda:0'))
        cudnn.benchmark = True
        print('    Total Conv and Linear Params: %.2fM' % (sum(p.weight.data.numel() for p in model.modules() if isinstance(p,nn.Linear) or isinstance(p,nn.Conv2d))/1000000.0))

        #Our config for skyline
        # criterion = nn.CrossEntropyLoss(reduction='mean')
        # optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)

        #Pruning config
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    else:

        print("==> creating model '{}'".format(args.arch))
        if args.arch.endswith('resnet'):
            model = models.__dict__[args.arch](
                        num_classes=num_classes,
                        depth=args.depth,
                    )
        else:
            model = models.__dict__[args.arch](num_classes=num_classes)

        model.cuda()
        max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated('cuda:0'))

        cudnn.benchmark = True
        print('    Total Conv and Linear Params: %.2fM' % (sum(p.weight.data.numel() for p in model.modules() if isinstance(p,nn.Linear) or isinstance(p,nn.Conv2d))/1000000.0))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    
    # Resume
    if args.presto_baseline:
        title = 'Presto Baseline'
    else:
        if args.dataset == 'cifar10':
            title = 'cifar-10-' + args.arch
        elif args.dataset == 'cifar100':
            title = 'cifar-100-' + args.arch
        else:
            title = 'tinyimagenet' + args.arch

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.save_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    save_checkpoint({'state_dict': model.state_dict()}, False, checkpoint=args.save_dir, filename='init.pth.tar')
    
    MAX_BATCH_IDX = args.max_batch_idx
    
    # Train and val
    for epoch in range(start_epoch, args.epochs):
            
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        print("MAXIMUM GPU USAGE : ", max_gpu_usage)

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        
        if MAX_BATCH_IDX == 0:
            test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        else:
            test_loss = 0
            test_acc = 0
        
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.save_dir)
        

    logger.close()
    
    print('Best acc:')
    print(best_acc)
    print("MAXIMUM GPU USAGE : ", max_gpu_usage)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    global max_gpu_usage
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # bar = Bar('Processing', max=len(trainloader))
    # print(args)
    
    MAX_BATCH_IDX = args.max_batch_idx
    
    for batch_idx, (_, inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if MAX_BATCH_IDX != 0:
            if batch_idx == MAX_BATCH_IDX:
                # bar.finish()
                return (losses.avg, top1.avg)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated('cuda:0'))
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        if args.dataset.endswith('rotate'):
            prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1, 2))
            top2.update(prec2.item(), inputs.size(0))
        else:
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top5.update(prec5.item(), inputs.size(0))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
    #     bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
    #                 batch=batch_idx + 1,
    #                 size=len(trainloader),
    #                 data=data_time.avg,
    #                 bt=batch_time.avg,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,
    #                 top1=top1.avg,
    #                 top5=top5.avg,
    #                 )
    #     bar.next()
    # bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (_, inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        if args.dataset.endswith('rotate'):
            prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1, 2))
            top2.update(prec2.item(), inputs.size(0))
        else:
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top5.update(prec5.item(), inputs.size(0))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top2: {top2: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top2=top2.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
