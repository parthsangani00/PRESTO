import os
import gc
import sys
import PIL
import copy
import time
import torch
import random
import pickle
import argparse
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, models, datasets

from myDatasets import *

def get_means(all_subset_scores, trainset):
    num_bins = all_subset_scores.shape[1]
    num_features = trainset.num_features
    ans = torch.zeros(num_bins, num_features)
    for bi in range(num_bins):
        bin_indices = all_subset_scores[:,bi] == 1
        bin_indices = torch.arange(bin_indices.shape[0])[bin_indices]
        ans[bi] = trainset.data[bin_indices,:].mean(0)
    return ans

# Is called for a bin
# takes in the model (for the bin) and the subset one hot vector (for the bin)
def BTrain(model, trainset, trainloader, subset, device, all_subset_scores, max_gpu_usage, lr = 0.001):

    TRAIN_BATCH_SIZE = 4096
    SUBSET_TRAIN_BATCH_SIZE = 256
    relu = nn.ReLU()

    num_epochs = 10

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs, 
                                                steps_per_epoch=len(trainloader))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000.0

    criterion = nn.CrossEntropyLoss(reduction='mean')

    for epoch in range(num_epochs):
        
        model.train()

        running_loss = 0.0
        count=0

        for _,(index,data,labels) in enumerate(trainloader):
            # data point in this bin
            subset_positions = subset[index] == 1
            if subset_positions.sum() == 0:
                continue

            data = data[subset_positions].to(device)
            labels = labels[subset_positions].to(device)

            max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))
            # print("GPU memory used in training : ", torch.cuda.memory_allocated(device))

            outputs = model(data.float())
            # loss = relu(1 - outputs.reshape(-1)*labels).sum()
            loss = criterion(outputs,labels)
            running_loss += (float(loss)*subset_positions.sum())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step()

            count+=subset_positions.sum()

        epoch_loss = running_loss / count
        print("running btrain, epoch {}, loss {}".format(epoch,epoch_loss))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    running_loss = 0.0

    return model, running_loss, max_gpu_usage

def BTrain_external_classifier(external_classifier, trainset, trainloader, subset_selection, device, models, max_gpu_usage, lr):

    NUM_BINS = len(models)

    for bi in range(NUM_BINS):
        models[bi].eval()

    external_classifier.train()

    num_epochs = 10
    optimizer = torch.optim.Adam(external_classifier.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs, 
                                                steps_per_epoch=len(trainloader))

    best_exc_wts = copy.deepcopy(external_classifier.state_dict())
    best_loss = 1000000.0

    criterion = nn.CrossEntropyLoss(reduction='mean')

    for epoch in range(num_epochs):
        
        running_loss = 0.0
        count=0

        for _,(index,data,labels) in enumerate(trainloader):

            data = data.to(device)
            labels = labels.to(device)
            exc_inputs = torch.zeros(len(data),NUM_BINS*trainset.num_classes).to(device)

            max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))
            # print("GPU memory used in training : ", torch.cuda.memory_allocated(device))

            for en,bmodel in enumerate(models):
                predictions = bmodel(data.float())
                exc_inputs[:,trainset.num_classes*en:trainset.num_classes*en+trainset.num_classes] = predictions

            outputs = external_classifier(exc_inputs)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step()

            running_loss += (loss*len(data))
            count += len(data)

        epoch_loss = running_loss / count
        print("running btrain, epoch {}, loss {}".format(epoch,epoch_loss))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_exc_wts = copy.deepcopy(external_classifier.state_dict())

    external_classifier.load_state_dict(best_exc_wts)

    # print("GPU memory used before m-score calc : ", torch.cuda.memory_allocated(device))
    
    all_scores = torch.zeros((len(trainset), NUM_BINS))
    all_scores = all_scores.to(device)

    for target in range(trainset.num_classes):

        # print("GPU memory used : ", torch.cuda.memory_allocated(device), " target = ", target)

        # index = [idx where trainset.labels[idx]==target]
        index = trainset.labels == target
        # print("Target ", target, " : ", index.sum()) # 5000
        # print("GPU memory used before data : ", torch.cuda.memory_allocated(device))
        data = trainset.data[index].to(device) # 5000,64 or 5000,128,8,8
        # print("GPU memory used after data, before exc_inputs : ", torch.cuda.memory_allocated(device))
        exc_inputs = torch.zeros(len(data),NUM_BINS*trainset.num_classes).to(device) # 5000, 40
        # print("GPU memory used after exc_inputs : ", torch.cuda.memory_allocated(device))

        max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))
        # print("GPU memory used : ", torch.cuda.memory_allocated(device))

        for en,bmodel in enumerate(models):
            # print("GPU memory used before line 1 : ", torch.cuda.memory_allocated(device))
            with torch.no_grad():
                predictions = bmodel(data.float())
            # print("GPU memory used before line 2 : ", torch.cuda.memory_allocated(device))
            exc_inputs[:,trainset.num_classes*en:trainset.num_classes*en+trainset.num_classes] = predictions

        # external_classifier weight is of size (10, 40). we use target and bin slice to get wts

        for bi in range(NUM_BINS):
            # print("GPU memory used before wts : ", torch.cuda.memory_allocated(device))
            wts = external_classifier.model[0].weight[target, bi*trainset.num_classes:(bi+1)*trainset.num_classes].to(device) # 10
            # print("GPU memory used after wts, before data slice : ", torch.cuda.memory_allocated(device))
            data_slice = exc_inputs[:, bi*trainset.num_classes:(bi+1)*trainset.num_classes].to(device) # 5000, 10

            max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))
            # print("GPU memory used : ", torch.cuda.memory_allocated(device), " target = ", target, " bin = ", bi)

            all_scores[:,bi][index] = torch.matmul(data_slice, wts) # 5000, 1

        # print("GPU memory used : ", torch.cuda.memory_allocated(device), " target = ", target)

    all_scores = all_scores.to("cpu")

    return external_classifier, all_scores, max_gpu_usage

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode='trunc')
    return tuple(reversed(out))

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_state(args, checkpoint_path, device, models, external_classifier, trainset, trainloader, clustering_models, NUM_BINS, LEE_WAY):
    if args.resume:
        model_state = torch.load(checkpoint_path + 'state.pth', map_location = device)['state']
        if (not args.state == -1):
            model_state = args.state
        print('Loading checkpoint at model state {}'.format(model_state), flush=True)
        dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state), map_location = device)
        pre_e = dic['e']
        for bi in range(NUM_BINS):
            models[bi].load_state_dict(dic['model_{}'.format(bi)])
        try:
            avg_bin_losses = dic['avg_bin_losses']
        except:
            avg_bin_losses = []
        subset_selection = dic['subset_selection']
        subset_selection = subset_selection.to("cpu")
        external_classifier.load_state_dict(dic['external_classifier'])
        
        print('Resuming Training after {} epochs'.format(pre_e))
        all_scores = dic['m_scores']
    else:
        model_state = 0
        pre_e =0
        losses = []
        avg_bin_losses = []
        subset_selection = torch.zeros(len(trainset), NUM_BINS)
        all_scores = torch.zeros(len(trainset), NUM_BINS)

        if not args.clustering_baseline:
            presto_file = 'presto_subset_selection_{}_{}_bins.pth'.format(args.dataset, NUM_BINS)

            if os.path.isfile(presto_file):
                dic = torch.load(presto_file, map_location = device)
                subset_selection = dic['initial_subset_selection']
                # If using MLR subset selection for CIFAR100 rotated on K=2, uncomment below.
                #subset_selection = dic['subset_selection'].cpu()
                temp = torch.zeros(NUM_BINS)
                for i in range(len(trainset)):
                    idx = torch.argmax(torch.Tensor(subset_selection[i]))
                    temp[idx] = temp[idx] + 1
                print("Initial bin wise allocation : ", temp.detach().cpu().numpy())
            else:
                if args.cluster_init is not None:
                    if args.cluster_init == "EqKMeans":
                        subset_selection = clustering_models[args.cluster_init](trainset.data, NUM_BINS, LEE_WAY)
                    else:
                        subset_selection = clustering_models[args.cluster_init](trainset.data, NUM_BINS)
                    cluster_subset_selection = np.copy(subset_selection)
                    temp = torch.zeros(NUM_BINS)
                    for i in range(len(trainset)):
                        idx = torch.argmax(torch.Tensor(subset_selection[i]))
                        temp[idx] = temp[idx] + 1
                    print("Initial bin wise allocation : ", temp.detach().cpu().numpy())
                else:
                    print('Using random initialization')
                    for i in range(len(trainset)):
                        temp = np.random.choice(NUM_BINS)
                        subset_selection[i, temp] = 1

                dic = {}
                dic['initial_subset_selection'] = subset_selection
                torch.save(dic, presto_file)
        else:
            try:
                assert args.cluster_init is not None
            except:
                print('Need to specify clustering method via args.cluster_init to use clustering baseline')

            subset_file = 'clustering_subset_selection_{}_{}_{}_bins.pth'.format(args.cluster_init, args.dataset, NUM_BINS)
            if os.path.isfile(subset_file):
                dic = torch.load(subset_file, map_location = device)
                subset_selection = dic[subset_file]
                temp = torch.zeros(NUM_BINS)
                for i in range(len(trainset)):
                    idx = torch.argmax(torch.Tensor(subset_selection[i]))
                    temp[idx] = temp[idx] + 1
                print("Initial bin wise allocation : ", temp.detach().cpu().numpy())
            else:
                if args.cluster_init is not None:
                    print('Using {} initialization'.format(args.cluster_init))
                    if args.cluster_init == "EqKMeans":
                        subset_selection = clustering_models[args.cluster_init](trainset.data, NUM_BINS, LEE_WAY)
                    elif args.cluster_init == "Manual":
                        subset_selection = clustering_models[args.cluster_init](trainset.data, trainset.labels, NUM_BINS)
                    else:
                        subset_selection = clustering_models[args.cluster_init](trainset.data, NUM_BINS)
                    cluster_subset_selection = np.copy(subset_selection)
                    temp = torch.zeros(NUM_BINS)
                    for i in range(len(trainset)):
                        idx = torch.argmax(torch.Tensor(subset_selection[i]))
                        temp[idx] = temp[idx] + 1
                    print("Initial bin wise allocation : ", temp.detach().cpu().numpy())
                else:
                    print('Using random initialization')
                    for i in range(len(trainset)):
                        temp = np.random.choice(NUM_BINS)
                        subset_selection[i, temp] = 1

                dic = {}
                dic[subset_file] = subset_selection
                torch.save(dic, subset_file)
    return model_state, pre_e, models, external_classifier, avg_bin_losses, subset_selection, all_scores


def BTrain_external_classifier_batched(external_classifier, trainset, trainloader, subset_selection, device, models, max_gpu_usage, lr):

    NUM_BINS = len(models)

    for bi in range(NUM_BINS):
        models[bi].eval()

    external_classifier.train()

    num_epochs = 10
    optimizer = torch.optim.Adam(external_classifier.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs, 
                                                steps_per_epoch=len(trainloader))

    best_exc_wts = copy.deepcopy(external_classifier.state_dict())
    best_loss = 1000000.0

    criterion = nn.CrossEntropyLoss(reduction='mean')

    for epoch in range(num_epochs):
        
        running_loss = 0.0
        count=0

        for _,(index,data,labels) in enumerate(trainloader):

            data = data.to(device)
            labels = labels.to(device)
            exc_inputs = torch.zeros(len(data),NUM_BINS*trainset.num_classes).to(device)

            max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))
            # print("GPU memory used in training : ", torch.cuda.memory_allocated(device))

            for en,bmodel in enumerate(models):
                predictions = bmodel(data.float())
                exc_inputs[:,trainset.num_classes*en:trainset.num_classes*en+trainset.num_classes] = predictions

            outputs = external_classifier(exc_inputs)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step()

            running_loss += (loss*len(data))
            count += len(data)

        epoch_loss = running_loss / count
        print("running btrain, epoch {}, loss {}".format(epoch,epoch_loss))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_exc_wts = copy.deepcopy(external_classifier.state_dict())

    external_classifier.load_state_dict(best_exc_wts)

    # print("GPU memory used before m-score calc : ", torch.cuda.memory_allocated(device))
    
    all_scores = torch.zeros((len(trainset), NUM_BINS))
    all_scores = all_scores.to(device)

    for target in range(trainset.num_classes):

        # print("GPU memory used : ", torch.cuda.memory_allocated(device), " target = ", target)

        # index = [idx where trainset.labels[idx]==target]
        index = trainset.labels == target
        # print("Target ", target, " : ", index.sum()) # 5000
        # print("GPU memory used before data : ", torch.cuda.memory_allocated(device))
        # data = trainset.data[index].to(device) # 5000,64 or 5000,128,8,8
        # print("GPU memory used after data, before exc_inputs : ", torch.cuda.memory_allocated(device))
        exc_inputs = torch.zeros(len(trainset.data[index]),NUM_BINS*trainset.num_classes).to(device) # 5000, 40
        # print("GPU memory used after exc_inputs : ", torch.cuda.memory_allocated(device))

        max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))
        # print("GPU memory used : ", torch.cuda.memory_allocated(device))

        num_iters = int(len(trainset.data[index]) / 2500) + 1

        for abc in range(num_iters):

            for en,bmodel in enumerate(models):
                # print("GPU memory used before line 1 : ", torch.cuda.memory_allocated(device))

                if abc == num_iters-1:
                    data = trainset.data[index][abc*2500:].to(device)
                else:
                    data = trainset.data[index][abc*2500:abc*2500+2500].to(device)

                max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

                with torch.no_grad():
                    predictions = bmodel(data.float())
                    
                # print("GPU memory used before line 2 : ", torch.cuda.memory_allocated(device))
                if abc == num_iters-1:
                    exc_inputs[abc*2500:,trainset.num_classes*en:trainset.num_classes*en+trainset.num_classes] = predictions
                else:
                    exc_inputs[abc*2500:abc*2500+2500,trainset.num_classes*en:trainset.num_classes*en+trainset.num_classes] = predictions

        # external_classifier weight is of size (10, 40). we use target and bin slice to get wts

        for bi in range(NUM_BINS):
            # print("GPU memory used before wts : ", torch.cuda.memory_allocated(device))
            wts = external_classifier.model[0].weight[target, bi*trainset.num_classes:(bi+1)*trainset.num_classes].to(device) # 10
            # print("GPU memory used after wts, before data slice : ", torch.cuda.memory_allocated(device))
            data_slice = exc_inputs[:, bi*trainset.num_classes:(bi+1)*trainset.num_classes].to(device) # 5000, 10

            max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))
            # print("GPU memory used : ", torch.cuda.memory_allocated(device), " target = ", target, " bin = ", bi)

            all_scores[:,bi][index] = torch.matmul(data_slice, wts) # 5000, 1

        # print("GPU memory used : ", torch.cuda.memory_allocated(device), " target = ", target)

    all_scores = all_scores.to("cpu")

    return external_classifier, all_scores, max_gpu_usage
