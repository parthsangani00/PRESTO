import os
import gc
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
from sklearn.cluster import *
from k_means_constrained import KMeansConstrained

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


os.environ['display'] = 'localhost:14.0'

def EqKMeans(data, N_BINS, LEE_WAY):
    groundData = data
    dim0 = groundData.shape[0]
    dim1 = groundData.shape[1]
    dim2 = groundData.shape[2]
    dim3 = groundData.shape[3]
    TRAIN_SIZE = len(groundData)
    PART_SIZE = TRAIN_SIZE//N_BINS
    clustering = KMeansConstrained(n_clusters=N_BINS, size_min=PART_SIZE-LEE_WAY, size_max=PART_SIZE+LEE_WAY, random_state=0)
    since = time.time()
    print("Running {}".format('EqKMeans'))
    clustering.fit(groundData.reshape((dim0, dim1*dim2*dim3)))
    labels = clustering.labels_
    subset_selection = np.zeros((TRAIN_SIZE,N_BINS))
    for i,label in enumerate(labels):
        subset_selection[i,label] = 1
    assert np.sum(subset_selection) == TRAIN_SIZE
    print("Time : ", time.time()-since)
    return subset_selection

def KMeansPP(data, N_BINS):
    groundData = data
    dim0 = groundData.shape[0]
    dim1 = groundData.shape[1]
    dim2 = groundData.shape[2]
    dim3 = groundData.shape[3]
    TRAIN_SIZE = len(groundData)
    clustering = KMeans(N_BINS, init='k-means++', random_state=0)
    since = time.time()
    print("Running {}".format('KMeans++'))
    clustering.fit(groundData.reshape((dim0, dim1*dim2*dim3)))
    labels = clustering.labels_
    subset_selection = np.zeros((TRAIN_SIZE,N_BINS))
    for i,label in enumerate(labels):
        subset_selection[i,label] = 1
    assert np.sum(subset_selection) == TRAIN_SIZE
    print("Time : ", time.time()-since)
    return subset_selection

def Agglomerative(data, N_BINS):
    groundData = data
    dim0 = groundData.shape[0]
    dim1 = groundData.shape[1]
    dim2 = groundData.shape[2]
    dim3 = groundData.shape[3]
    TRAIN_SIZE = len(groundData)
    clustering = AgglomerativeClustering(N_BINS)
    since = time.time()
    print("Running {}".format('Agglomerative Clustering'))
    clustering.fit(groundData.reshape((dim0, dim1*dim2*dim3)))
    labels = clustering.labels_
    subset_selection = np.zeros((TRAIN_SIZE,N_BINS))
    for i,label in enumerate(labels):
        subset_selection[i,label] = 1
    assert np.sum(subset_selection) == TRAIN_SIZE
    print("Time : ", time.time()-since)
    return subset_selection

def Manual(data, labels, N_BINS):
    groundData = data
    TRAIN_SIZE = len(groundData)
    since = time.time()
    print("Running {}".format('Manual'))
    subset_selection = np.zeros((TRAIN_SIZE,N_BINS))
    
    index = labels == 0
    subset_selection[:,0][index] = 1
    index = labels == 1
    subset_selection[:,0][index] = 1

    index = labels == 2
    subset_selection[:,1][index] = 1
    index = labels == 3
    subset_selection[:,1][index] = 1

    index = labels == 4
    subset_selection[:,2][index] = 1
    index = labels == 5
    subset_selection[:,2][index] = 1

    index = labels == 6
    subset_selection[:,3][index] = 1
    index = labels == 7
    subset_selection[:,3][index] = 1

    for i in range(len(data)):
        if labels[i]==8 or labels[i]==9:
            temp = np.random.choice(N_BINS)
            subset_selection[i, temp] = 1

    assert np.sum(subset_selection) == TRAIN_SIZE
    print("Time : ", time.time()-since)
    return subset_selection

def GMM(myData, N_BINS): #myData :-> Dataset object
    # groundData = myData.data.numpy()
    groundData = myData
    dim0 = groundData.shape[0]
    dim1 = groundData.shape[1]
    dim2 = groundData.shape[2]
    dim3 = groundData.shape[3]
    TRAIN_SIZE = len(groundData)
    gm = GaussianMixture(n_components=N_BINS, random_state=0)
    since = time.time()
    print("Running {}".format('GMM'))
    gm.fit(groundData.reshape((dim0, dim1*dim2*dim3)))
    labels = gm.predict(groundData.reshape((dim0, dim1*dim2*dim3)))
    subset_selection = np.zeros((TRAIN_SIZE,N_BINS))
    for i,label in enumerate(labels):
        subset_selection[i,label] = 1
    assert np.sum(subset_selection) == TRAIN_SIZE
    print("Time : ", time.time()-since)
    return subset_selection

def BGM(myData, N_BINS): #myData :-> Dataset object
    # groundData = myData.data.numpy()
    groundData = myData
    dim0 = groundData.shape[0]
    dim1 = groundData.shape[1]
    dim2 = groundData.shape[2]
    dim3 = groundData.shape[3]
    TRAIN_SIZE = len(groundData)
    gm = BayesianGaussianMixture(n_components=N_BINS, random_state=0)
    since = time.time()
    print("Running {}".format('BGM'))
    gm.fit(groundData.reshape((dim0, dim1*dim2*dim3)))
    labels = gm.predict(groundData.reshape((dim0, dim1*dim2*dim3)))
    subset_selection = np.zeros((TRAIN_SIZE,N_BINS))
    for i,label in enumerate(labels):
        subset_selection[i,label] = 1
    assert np.sum(subset_selection) == TRAIN_SIZE
    print("Time : ", time.time()-since)
    return subset_selection

def validate_cluster(ss, labels, N_BINS):
        # labels = cifar['test_labels']
        for cl in range(N_BINS):
            x = labels[ss[:,cl]==1]
            unique, counts = np.unique(x, return_counts=True)

            print(cl) 
            print(np.asarray((unique, counts)))


clustering_models = {
    'EqKMeans':EqKMeans,
    'Manual':Manual,
    'KMeans++':KMeansPP,
    'Agglomerative':Agglomerative,
    'GMM':GMM,
    'BGM':BGM,
}