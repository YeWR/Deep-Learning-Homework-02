import pandas as pd
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn import datasets
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import loss
import utils
import tqdm
import weight as wt
import csv
import os
from scipy import stats
import seaborn as sns

def scatter(x, colors, file_name, scale=7):
    f = plt.figure(figsize=(226 / 15, 212 / 15))
    ax = plt.subplot(aspect='equal')
    x_source, x_target = x[colors == 1, :], x[colors == 0, :]
    ## draw data points in a two-demensions-space
    ax.scatter(x_source[:, 0], x_source[:, 1], lw=0, s=scale, color='red')
    ax.scatter(x_target[:, 0], x_target[:, 1], lw=0, s=scale, color='blue')
    # plt.xlim(-40, 40)
    # plt.ylim(-40, 40)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.axis('off')
    ax.axis('tight')
    f.savefig(file_name + "_scale_%d" % scale + ".jpg", bbox_inches='tight')
    plt.show()

def plot_TSNE(dset_loader, model, title):
    base_network = model["base_network"]

    base_network.train(False)
    source_dset = dset_loader['source']
    target_dset = dset_loader['test0']
    x_source = []
    x_target = []

    # source
    for test_img, test_label, _ in source_dset:
        features_test, outputs_test = base_network(test_img.cuda())

        for out in outputs_test.detach().cpu().numpy():
            x_source.append(out)

    # target
    for test_img, test_label, _ in target_dset:
        features_test, outputs_test = base_network(test_img.cuda())

        for out in outputs_test.detach().cpu().numpy():
            x_target.append(out)

    x_source = np.array(x_source)
    x_target = np.array(x_target)
    colors = np.array([1] * len(x_source) + [0] * len(x_target))
    all_features_np = np.concatenate((x_source, x_target))
    tsne_features = TSNE(random_state=20190129).fit_transform(all_features_np)
    for scale in range(6, 15):
        scatter(tsne_features, colors, 'res/' + title, scale)
