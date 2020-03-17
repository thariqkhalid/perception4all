"""
__author__ = "Thariq Khalid"
__copyright__ = "Copyright 2020, The Perception4all Project"
__credits__ = ["Thariq Khalid"]

__maintainer__ = "Thariq Khalid"
__email__ = "thariq.khalid@gmail.com"
__status__ = "Research and Development"

"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from networks import vanilla_cnn
import data_loader

debug = False

net = vanilla_cnn.Net()

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

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

net.conv1.register_forward_hook(get_activation('conv1'))


_,testloader,_ = data_loader.datasetL()
it = iter(testloader)
data, _ = next(it)
data.unsqueeze(0)
output = net(data)

act = activation['conv1'].squeeze()
fig, axarr = plt.subplots(act.size(0))
for idx in range(act.size(0)):
    axarr[idx].imshow(act[idx])


# Next step is to study the code from https://www.kaggle.com/sironghuang/understanding-pytorch-hooks
