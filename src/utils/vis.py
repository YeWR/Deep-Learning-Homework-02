import pandas as pd
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn import svm, datasets
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import utils
import tqdm
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

# confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def do_confusion_matrix(loader, model, title, file_name):
    classes = [str(i) for i in range(65)]

    y_test = []
    y_pred = []

    model.train(False)

    for test_img, test_label, _ in loader:
        _, outputs_test = model(test_img.cuda())

        output = outputs_test.detach().cpu().numpy()
        pred = np.Softmax(output)
        pred = np.argmax(pred)

        y_pred += pred.tolist()
        y_test += test_label.detach().cpu().numpy().tolist()

    plot_confusion_matrix(y_test, y_pred, classes=classes, normalize=True,
                          title=title)
    plt.savefig(file_name, bbox_inches='tight')


if __name__ == '__main__':
    pass
    # resume_path = '../../models/pretrained-0,bottleneck-0,augmentation-1,weight_init-1,opt_type-SGD,momentum-0.9,lr-0.006,batch-32,weight-1.0,debug-model_C_att'
    # resume_ckpt = torch.load(resume_path)