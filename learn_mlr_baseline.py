from logging.handlers import TimedRotatingFileHandler
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
# from myModels import MultiHidden, LinearNN, ConvBlocks, ConvBlocks_CIFAR10
from myModels import LinearNN, ConvBlocks, ConvBlocks_CIFAR10
from clustering import *
from arguments import myArgParse

# TODO: MLR runs into zero bin problem on CIFAR10 and CIFAR100. Fix.
args = myArgParse()
seed_torch(args.seed)
if args.cuda == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
print('Device:',device)

SUBSET_TRAIN_BATCH_SIZE = 256
lr = 1e-2
NUM_EPOCHS = 30
NUM_BINS = args.n_bins

SAVE_INTERVAL = 1

torch.autograd.set_detect_anomaly(True)

checkpoint_path = './' + args.ckpt + '/'
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

print('Checkpoint: {}'.format(checkpoint_path))

if args.dataset in ['cifar10-rotate', 'cifar10_rotate']:

    LEE_WAY = 100000

    trainset = CIFAR10_ROTATE(train = True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = True)
    testset = CIFAR10_ROTATE(train = False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = False)

    lrs = [lr for _ in range(NUM_BINS)]
    models = []
    for _ in range(NUM_BINS):
        models.append(ConvBlocks_CIFAR10(input_dim=trainset.num_features, hidden_dim1=8, hidden_dim2=16, output_dim=trainset.num_classes, is_downsample=True).to(device))

    external_classifier = LinearNN(input_dim = NUM_BINS*trainset.num_classes, output_dim = trainset.num_classes).to(device)

elif args.dataset in ['cifar100-rotate', 'cifar100_rotate']:

    LEE_WAY = 100000

    trainset = CIFAR100_ROTATE(train = True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = True)
    testset = CIFAR100_ROTATE(train = False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = False)

    lrs = [lr for _ in range(NUM_BINS)]
    models = []
    for _ in range(NUM_BINS):
        models.append(ConvBlocks_CIFAR10(input_dim=trainset.num_features, hidden_dim1=8, hidden_dim2=16, output_dim=trainset.num_classes, is_downsample=True).to(device))

    external_classifier = LinearNN(input_dim = NUM_BINS*trainset.num_classes, output_dim = trainset.num_classes).to(device)

elif args.dataset in ['stl-rotate', 'stl_rotate']:

    LEE_WAY = 100000

    trainset = STL10_ROTATE(train = True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = True)
    testset = STL10_ROTATE(train = False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=SUBSET_TRAIN_BATCH_SIZE, shuffle = False)

    lrs = [lr for _ in range(NUM_BINS)]
    models = []
    for _ in range(NUM_BINS):
        models.append(ConvBlocks_CIFAR10(input_dim=trainset.num_features, hidden_dim1=8, hidden_dim2=16, output_dim=trainset.num_classes, is_downsample=True).to(device))

    external_classifier = LinearNN(input_dim = NUM_BINS*trainset.num_classes, output_dim = trainset.num_classes).to(device)

max_gpu_usage = torch.cuda.memory_allocated(device)

regulariser = LinearNN(input_dim = 1, output_dim = 1) 
model_state, pre_e, models, external_classifier, avg_bin_losses, subset_selection, all_scores \
    = set_state(args, checkpoint_path, device, models, external_classifier, trainset, trainloader, clustering_models, NUM_BINS, LEE_WAY)
print('Starting Training')

def evaluate_external_classifier(models, external_classifier, lrs, subset_selection, trainset, trainloader, testset, testloader, max_gpu_usage, NUM_BINS, k=1, load_init_model=False, train=True):
    
    correct = 0
    total = 0
    external_classifier.eval()

    if train==True:

        '''for i in tqdm(range(len(trainset))):
        # for i in tqdm(range(2)):
        
            data = trainset.data[i,:].to(device)
            label = trainset.labels[i].to(device)'''
        for _,(index,data,labels) in enumerate(trainloader):

            data = data.to(device)
            label = labels.to(device)
            exc_inputs = torch.zeros(len(data),NUM_BINS*trainset.num_classes).to(device)

            # max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))
            # print("GPU memory used : ", torch.cuda.memory_allocated(device))

            for en,bmodel in enumerate(models):
                predictions = bmodel(data.float())
                exc_inputs[:,trainset.num_classes*en:trainset.num_classes*en+trainset.num_classes] = predictions

            outputs = external_classifier(exc_inputs)
            #m = nn.Softmax()
            #prob = m(outputs)
            #predicted_label = torch.argmax(prob)
            
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
        print('Training Accuracy : ',100*correct / total, '%')
        return 0

    elif train==False:

        '''for i in tqdm(range(len(testset))):
        # for i in tqdm(range(2)):
        
            data = testset.data[i,:].to(device)
            label = testset.labels[i].to(device)'''
        for _,(index,data,labels) in enumerate(testloader):

            data = data.to(device)
            label = labels.to(device)
            exc_inputs = torch.zeros(len(data),NUM_BINS*trainset.num_classes).to(device)

            # max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))
            # print("GPU memory used : ", torch.cuda.memory_allocated(device))

            for en,bmodel in enumerate(models):
                predictions = bmodel(data.float())
                exc_inputs[:,trainset.num_classes*en:trainset.num_classes*en+trainset.num_classes] = predictions

            outputs = external_classifier(exc_inputs)
            # m = nn.Softmax()
            # prob = m(outputs)
            # predicted_label = torch.argmax(prob)

            #if predicted_label==label:
            #    correct += 1
            
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
        print('Test Accuracy : ',100*correct / total, '%')
        return (100*correct / total)

def get_means():
    # num_features = trainset.num_features
    f0 = trainset.data.shape[0]
    f1 = trainset.data.shape[1]
    f2 = trainset.data.shape[2]
    f3 = trainset.data.shape[3]
    dataset = trainset.data.reshape((f0,f1*f2*f3))
    ans = torch.zeros(NUM_BINS, f1*f2*f3)
    for bi in range(NUM_BINS):
        bin_indices = subset_selection[:,bi] == 1
        bin_indices = torch.arange(bin_indices.shape[0])[bin_indices]
        ans[bi] = torch.Tensor(dataset[bin_indices,:].mean(0))
    return ans

def train_batch(bi, subset_scores, max_gpu_usage, to_train = True, bin = 0):
    """
    all_scores[:,bi] = train_batch(bi, subset_selection[:,bi], allmeans, bin = bi)
    PRESTO 5-11: For each bin, train_batch() calculates m_scores for that bin
    For a bin == S^, BTrain computes:
        w^ = model[bi], which is trained on bi, for 100 ep
        f^(S^) = fhat_whole[bi], trained model total loss on the points in bi
    BTrain_batch computes, a batch at a time:
        f^(S^\{i}) = fhat_partial, 
            train model on all points in bin bi except i,
            fhat_partial = total loss of this model on bin i (AC: including on point i?)
        m_scores[i,bi] for i in bi = alpha*(f^(S^)-f^(S^\{i}))
        m_scores[i,bi] for i not in bi = loss of model[bi] on i 
    
    """
    #####
    criterion = nn.CrossEntropyLoss(reduction='none')
    global models
    m_scores = torch.zeros((len(trainset),))
    if subset_scores.sum() == 0:
        print("Bin {} Empty".format(bi),flush = True)
        return m_scores

    # BTrain - it trains the bin-model on the data points in the bin before the epoch started
    if to_train:
        models[bi], _, max_gpu_usage = BTrain(models[bi], trainset, trainloader, subset_selection[:,bi], device, subset_selection, max_gpu_usage, lr = lrs[bi])
    
    all_indices = torch.arange(len(trainset)).long()
    bin_indices = all_indices
    num_bin_batches = int(np.ceil(len(bin_indices) / SUBSET_TRAIN_BATCH_SIZE))
    
    for i in tqdm(range(num_bin_batches), desc = "[ Training, Bin {} ".format(bin)):
        if i == num_bin_batches-1:
            index = bin_indices[i*SUBSET_TRAIN_BATCH_SIZE:]
        else:
            index = bin_indices[i*SUBSET_TRAIN_BATCH_SIZE:(i+1)*SUBSET_TRAIN_BATCH_SIZE]
        
        data = trainset.data[index].to(device)
        labels = trainset.labels[index].to(device)
        with torch.no_grad():
            outputs = models[bi](data)

        max_gpu_usage = max(max_gpu_usage, torch.cuda.memory_allocated(device))

        m_scores[index] = criterion(outputs,labels).to("cpu")

    return m_scores, max_gpu_usage

best_test_acc = -1
best_test_acc_epoch = -1
print("Max GPU Usage before training : ", max_gpu_usage)
for e in range(NUM_EPOCHS):
    start = time.time()
    torch.cuda.empty_cache()
    if pre_e > 0:
        pre_e -= 1
        continue
    print("GPU Usage at start of epoch : ",torch.cuda.memory_allocated(device))
    all_scores = torch.zeros(len(trainset), NUM_BINS)
    print('Training for epoch {}'.format(e+1), flush = True)

    all_scores = torch.zeros((len(trainset), NUM_BINS))

    for bi in range(NUM_BINS):
        torch.cuda.empty_cache()
        print('Training for bin {}'.format(bi), flush = True)
        if e==0:
            all_scores[:,bi], max_gpu_usage = train_batch(bi, subset_selection[:,bi], max_gpu_usage, to_train = False, bin = bi)
        else:
            all_scores[:,bi], max_gpu_usage = train_batch(bi, subset_selection[:,bi], max_gpu_usage, bin = bi)

    print("Max GPU Usage after BTrain : ", max_gpu_usage)
    torch.cuda.empty_cache()

    print('Training for external classifier', flush = True)
    if e!=0:
        external_classifier, _, max_gpu_usage = BTrain_external_classifier(external_classifier, trainset, trainloader, subset_selection, device, models, max_gpu_usage, lr)

    print("Max GPU Usage after BTrain_external_classifier : ", max_gpu_usage)

    subset_selection_clone = torch.clone(torch.Tensor(subset_selection))
    part_size = len(trainset) // NUM_BINS
    part_size += LEE_WAY
    all_scores2 = torch.zeros(len(trainset),NUM_BINS)
    for i in range(len(trainset)):
        for bi in range(NUM_BINS):
            all_scores2[i,bi] = all_scores[i,bi]

    if args.clustering_baseline or e>30:
        subset_selection = torch.Tensor(subset_selection)
    else:
        subset_selection = torch.zeros(len(trainset), NUM_BINS)

    bin_transfer = torch.zeros((NUM_BINS,NUM_BINS))

    bin_counts = torch.Tensor([0]*NUM_BINS)
    all_scores_cp2 = torch.clone(all_scores2)
    
    if args.clustering_baseline or e>30:
        pass
    else:
        print("Re-assigning bins based on updated m-scores", flush=True)
        for i in tqdm(range(len(trainset))):
            ind = unravel_index(torch.argmin(all_scores_cp2),all_scores_cp2.shape)
            old_idx = torch.argmax(subset_selection_clone[ind[0]])
            bin_transfer[old_idx, ind[1]] = bin_transfer[old_idx, ind[1]] + 1
            subset_selection[ind[0],ind[1]] = 1
            all_scores_cp2[ind[0],:]=99999
            bin_counts[ind[1]]+=1
            if(bin_counts[ind[1]]==part_size):
                all_scores_cp2[:,ind[1]]=99999

    print("Bin counts : ", bin_counts.detach().cpu().numpy())
    print("Bin transfer : ", bin_transfer.detach().cpu().numpy())

    bin_wise_class_count = torch.zeros((NUM_BINS, trainset.num_classes))
    for i in tqdm(range(len(trainset))):
        bin = torch.argmax(subset_selection[i,:])
        label = trainset.labels[i]
        bin_wise_class_count[bin,label] = bin_wise_class_count[bin,label] + 1
    
    for bi in range(NUM_BINS):
        print("Bin ", bi, " : ", bin_wise_class_count[bi,:])
    
    evaluate_external_classifier(models, external_classifier, lrs, subset_selection, trainset, trainloader, testset, testloader, max_gpu_usage,\
    NUM_BINS, load_init_model=False, train=True)
    test_acc = evaluate_external_classifier(models, external_classifier, lrs, subset_selection, trainset, trainloader, testset, testloader, max_gpu_usage,\
    NUM_BINS, load_init_model=False, train=False)

    best_acc = False
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_test_acc_epoch = e+1

        best_acc = True

    print("Max GPU Usage at epoch ", (e+1), " : ", max_gpu_usage)

    print("GPU Usage at end of epoch : ",torch.cuda.memory_allocated(device))

    dic = {}
    dic['e'] = e+1
    for bi in range(NUM_BINS):
        dic['model_{}'.format(bi)] = models[bi].state_dict()
    dic['subset_selection'] = subset_selection
    dic['m_scores'] = all_scores
    dic['old_subset_selection'] = subset_selection_clone
    dic['external_classifier'] = external_classifier.state_dict()

    if best_acc:
        # Save the subset selection
        subset_file = 'clustering_subset_selection_MLR_{}_{}_bins.pth'.format(args.dataset, NUM_BINS)
        dic[subset_file] = dic['subset_selection']
        torch.save(dic, subset_file)

    if (e+1) % SAVE_INTERVAL == 0:
        torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
        torch.save({'state': model_state}, checkpoint_path + 'state.pth')
        model_state += 1
        print('Saving model after {} Epochs'.format(e+1))

    print(f"Epoch {e + 1} took {time.time() - start} seconds")

print("Best Test Accuracy : ", best_test_acc, " %")
print("Best Test Accuracy at epoch : ", best_test_acc_epoch)
