import os
import gc
from re import M
import sys
import PIL
import time
import torch
import random
import pickle
import sklearn
import argparse
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, models, datasets
from sklearn.cluster import KMeans

os.environ['display'] = 'localhost:14.0'

from myDatasets import *
from helperFunctionsJoint import *
from myModels import LinearNN, ConvBlocks, ConvBlocks_CIFAR10
# from clustering import *
from arguments import myArgParse

parser = myArgParse(parse_args=False)
# Baseline hidden dim arguments. Defaults taken for the rotated datasets case.
parser.add_argument('--hidden_dim_1', type = int, default = 24)
parser.add_argument('--hidden_dim_2', type = int, default = 32)
parser.add_argument('--hidden_dim_3', type = int, default = 0)


if __name__ == "__main__":
    args = parser.parse_args()

    seed_torch(args.seed)
    if args.cuda == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    print('Device:',device)

    SUBSET_TRAIN_BATCH_SIZE = 256
    NUM_EPOCHS = 15

    SAVE_INTERVAL = 1

    T = 5 # KD hyperparameter
    KD_LAMBDA = 0.9 # KD hyperparameter

    torch.autograd.set_detect_anomaly(True)

    max_gpu_usage = torch.cuda.memory_allocated(device)

    checkpoint_path = './' + args.ckpt + '/'
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    print('Checkpoint: {}'.format(checkpoint_path))

    if args.dataset in ['cifar10-rotate', 'cifar10_rotate']:

        NUM_EPOCHS = 10 # cuda 1

        trainset = CIFAR10_ROTATE(train = True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = True)
        testset = CIFAR10_ROTATE(train = False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = False)

    elif args.dataset in ['cifar100-rotate', 'cifar100_rotate']:

        NUM_EPOCHS = 10 # cuda 1

        trainset = CIFAR100_ROTATE(train = True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = True)
        testset = CIFAR100_ROTATE(train = False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = False)

    elif args.dataset in ['stl-rotate', 'stl_rotate']:

        NUM_EPOCHS = 10 # cuda 1

        trainset = STL10_ROTATE(train = True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = True)
        testset = STL10_ROTATE(train = False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = False)


    def train_skyline(trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline, hidden_dim1, hidden_dim2, hidden_dim3):
        
        seed_torch(args.seed)
        if 'rotate' in args.dataset or args.dataset == 'cifar10':
            classifier = ConvBlocks_CIFAR10(input_dim=trainset.num_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=trainset.num_classes, is_downsample=True).to(device)
        else:
            classifier = ConvBlocks(input_dim=trainset.num_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3, output_dim=trainset.num_classes, is_downsample=True).to(device)
        classifier.train()

        max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

        num_epochs = NUM_EPOCHS
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,1e-3, epochs=num_epochs, 
                                                    steps_per_epoch=len(trainloader))

        best_c_wts = copy.deepcopy(classifier.state_dict())
        best_loss = 1000000.0

        criterion = nn.CrossEntropyLoss(reduction='mean')

        losses = []

        for epoch in range(num_epochs):

            print("Running epoch ",epoch,flush=True)

            running_loss = 0.0
            count = 0

            for _,(index,data,labels) in enumerate(trainloader):

                data = data.to(device)
                labels = labels.to(device)

                max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

                outputs = classifier(data)
                loss = criterion(outputs,labels)

                max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sched.step()

                max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

                running_loss += (loss*len(data))
                count += len(data)

            epoch_loss = running_loss / count
            losses.append(epoch_loss)
            print("running skyline, epoch {}, loss {}".format(epoch,epoch_loss))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_c_wts = copy.deepcopy(classifier.state_dict())

        classifier.load_state_dict(best_c_wts)

        max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

        correct = 0
        total = 0
        for _,(index,data,labels) in enumerate(trainloader):

            data = data.to(device)
            label = labels.to(device)

            outputs = classifier(data)

            max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

            #m = nn.Softmax()
            #prob = m(outputs)
            #predicted_label = torch.argmax(prob)
                
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

        print('Skyline Training Accuracy : ',100*correct / total, '%')

        correct = 0
        total = 0
        for _,(index,data,labels) in enumerate(testloader):

            data = data.to(device)
            label = labels.to(device)

            outputs = classifier(data)

            max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

            #m = nn.Softmax()
            #prob = m(outputs)
            #predicted_label = torch.argmax(prob)
                
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
                

        print('Skyline Test Accuracy : ',100*correct / total, '%')

        dic={}
        dic['skyline'] = classifier.state_dict()
        torch.save(dic, checkpoint_path + 'DGKD_skyline.pth')


        # ypoints = np.array(losses)
        # plt.plot(ypoints)
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.savefig('skyline_KD.png')
        # plt.close()

        return classifier, max_gpu_usage


    def train_baseline(teacher, ta, trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline, hidden_dim1, hidden_dim2, hidden_dim3, get_param_count=False):
        
        seed_torch(args.seed)
        if 'rotate' in args.dataset or args.dataset == 'cifar10':
            student = ConvBlocks_CIFAR10(input_dim=trainset.num_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=trainset.num_classes, is_downsample=True).to(device)
        else:
            student = ConvBlocks(input_dim=trainset.num_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3, output_dim=trainset.num_classes, is_downsample=True).to(device)
        student.train()

        if get_param_count:
            return student

        max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

        num_epochs = NUM_EPOCHS
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,1e-3, epochs=num_epochs, 
                                                    steps_per_epoch=len(trainloader))

        best_c_wts = copy.deepcopy(student.state_dict())
        best_loss = 1000000.0

        criterion = nn.CrossEntropyLoss(reduction='mean')

        losses = []

        for epoch in range(num_epochs):

            print("Running epoch ",epoch,flush=True)

            running_loss = 0.0
            count = 0

            for _,(index,data,labels) in enumerate(trainloader):

                data = data.to(device)
                labels = labels.to(device)

                max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

                outputs = student(data)
                loss_CE = criterion(outputs,labels)

                max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

                with torch.no_grad():
                    teacher_outputs = teacher(data)
                    ta_outputs = ta(data)

                loss_KL = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1))
                loss_KL += nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(ta_outputs / T, dim=1))

                loss = (1 - KD_LAMBDA)*loss_CE + KD_LAMBDA*loss_KL*T*T

                max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sched.step()

                max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

                running_loss += (loss*len(data))
                count += len(data)

            epoch_loss = running_loss / count
            losses.append(epoch_loss)
            print("running baseline, epoch {}, loss {}".format(epoch,epoch_loss))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_c_wts = copy.deepcopy(student.state_dict())

        student.load_state_dict(best_c_wts)

        max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

        correct = 0
        total = 0
        for _,(index,data,labels) in enumerate(trainloader):

            data = data.to(device)
            label = labels.to(device)

            outputs = student(data)

            max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

            #m = nn.Softmax()
            #prob = m(outputs)
            #predicted_label = torch.argmax(prob)
                
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

        print('Baseline Training Accuracy : ',100*correct / total, '%')

        correct = 0
        total = 0
        for _,(index,data,labels) in enumerate(testloader):

            data = data.to(device)
            label = labels.to(device)

            outputs = student(data)

            max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

            #m = nn.Softmax()
            #prob = m(outputs)
            #predicted_label = torch.argmax(prob)
                
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
                
        print('Baseline Test Accuracy : ',100*correct / total, '%')

        dic={}
        dic['student'] = student.state_dict()
        torch.save(dic, checkpoint_path + 'DGKD_student.pth')

        # ypoints = np.array(losses)
        # plt.plot(ypoints)
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.savefig('baseline_KD.png')
        # plt.close()

        return max_gpu_usage


    if 'rotate' in args.dataset or args.dataset == 'cifar10':
        if args.get_param_count:
            student = train_baseline(None, None, trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline=True, hidden_dim1=args.hidden_dim_1, hidden_dim2=args.hidden_dim_2, hidden_dim3=args.hidden_dim_3,
                                     get_param_count=True)
            params = sum(p.numel() for p in student.parameters() if p.requires_grad)
            print("The number of parameters is -: ", params)
            sys.exit(0)

        teacher, max_gpu_usage = train_skyline(trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline=False, hidden_dim1=256, hidden_dim2=512, hidden_dim3=0) #Teacher
        ta, max_gpu_usage = train_skyline(trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline=False, hidden_dim1=64, hidden_dim2=128, hidden_dim3=0) #TA
        max_gpu_usage = train_baseline(teacher, ta, trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline=True, hidden_dim1=args.hidden_dim_1, hidden_dim2=args.hidden_dim_2, hidden_dim3=args.hidden_dim_3)
    else:
        teacher, max_gpu_usage = train_skyline(trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline=False, hidden_dim1=128, hidden_dim2=256, hidden_dim3=512) #Teacher
        ta, max_gpu_usage = train_skyline(trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline=False, hidden_dim1=32, hidden_dim2=64, hidden_dim3=128) #TA
        max_gpu_usage = train_baseline(teacher, ta, trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline=True, hidden_dim1=12, hidden_dim2=16, hidden_dim3=32)


    print("Max GPU Usage : ", max_gpu_usage)

    # if not args.clustering_baseline:

    #     teacher, max_gpu_usage = train_skyline(trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline=False, hidden_dim1=128, hidden_dim2=256, hidden_dim3=512) #Teacher
    #     ta, max_gpu_usage = train_skyline(trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline=False, hidden_dim1=32, hidden_dim2=64, hidden_dim3=128) #TA
    #     max_gpu_usage = train_baseline(teacher, ta, trainset, trainloader, testset, testloader, max_gpu_usage, is_baseline=True, hidden_dim1=12, hidden_dim2=16, hidden_dim3=32)

    #     print("Max GPU Usage : ", max_gpu_usage)

    # else:

    #     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #     timings=[]
    #     _count=0

    #     test_max_gpu_usage = torch.cuda.memory_allocated(device)
    #     hidden_dim1 = 12
    #     hidden_dim2 = 16
    #     hidden_dim3 = 24
    #     student = ConvBlocks(input_dim=trainset.num_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3, output_dim=trainset.num_classes, is_downsample=True).to(device)
    #     student.load_state_dict(torch.load(checkpoint_path + 'student.pth')['student'])
    #     test_max_gpu_usage = max(test_max_gpu_usage, torch.cuda.memory_allocated(device))

    #     correct = 0
    #     total = 0
    #     for idx,(index,data,labels) in enumerate(testloader):

    #         data = data.to(device)
    #         label = labels.to(device)

    #         # _count += len(data)
    #         _count += 1

    #         starter.record()
    #         outputs = student(data)
    #         ender.record()
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         # print(curr_time, len(data))
    #         timings.append(curr_time)

    #         # print(idx, " ", torch.cuda.memory_allocated(device))
    #         test_max_gpu_usage = max(test_max_gpu_usage, torch.cuda.memory_allocated(device))

    #         # max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

    #         #m = nn.Softmax()
    #         #prob = m(outputs)
    #         #predicted_label = torch.argmax(prob)
                
    #         _, predicted = outputs.max(1)
    #         total += label.size(0)
    #         correct += predicted.eq(label).sum().item()
                
    #     print('Student Test Accuracy : ',100*correct / total, '%')
    #     print('GPU Usage : ', test_max_gpu_usage)
    #     # print('Inference Time : ', sum(timings)/_count)
    #     print('Inference Time : ', sum(timings[1:])/(_count-1))