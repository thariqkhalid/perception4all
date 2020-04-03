"""
__author__ = "Thariq Khalid"
__copyright__ = "Copyright 2020, The Perception4all Project"
__credits__ = ["Thariq Khalid"]

__maintainer__ = "Thariq Khalid"
__email__ = "thariq.khalid@gmail.com"
__status__ = "Research and Development"

"""
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import itertools
import numpy as np

from networks import vanilla_cnn
import data_loader
from config import *
from sklearn.metrics import confusion_matrix

debug = False

net = vanilla_cnn.Net()
net.load_state_dict(torch.load('experiments/models/working/cifar_net.pth'))

if debug:
    #Let's see the network contents
    print("the network contents")
    print(net)

    #Let's see the layers
    print("Neural Network layers 1 by 1:")
    for layer in enumerate(net.modules()):
        print(layer[1])

    #Let's see the parameters
    print("Neural Network parameters")
    parameters = list(net.parameters())
    for p in parameters:
        print(p.shape)


# Visualize feature maps
activation = {}

def get_activation(name):
    def hook(model, input, output): # standard way
        activation[name] = output.detach()
    return hook

# Also refer to the code from https://www.kaggle.com/sironghuang/understanding-pytorch-hooks
def visualize_layer(layer_name):
    net.conv1.register_forward_hook(get_activation('fc3'))

    _, testloader, _ = data_loader.datasetL()
    it = iter(testloader)
    data, labels = next(it)
    output = net(data)

    act = activation['fc3']

    fig, axis = plt.subplots(act.size(1), act.size(0))

    for b in range(act.size(0)):
        axis[0,b].set_title(classes[labels[b].numpy()])
        for idx in range(act.size(1)):
            axis[idx, b].imshow(act[b][idx])

    plt.show()


def plot_confusion_matrix(cm, target_names, normalize, cmap, title):
    # this code is from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


if __name__ == "__main__":
    _, testloader, _ = data_loader.datasetL()

    full_labels = []
    full_predicted = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            for i in range(labels.size(0)):
                full_predicted.append(int(predicted[i].numpy()))
                full_labels.append(int(labels[i].numpy()))


    cm = confusion_matrix(full_labels, full_predicted)
    plot_confusion_matrix(cm, target_names= list(classes_mnist), normalize=False, cmap=None, title="Confusion Matrix")
