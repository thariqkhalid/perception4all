"""
__author__ = "Afnan Qalas Balghaith, Thariq Khalid"
__copyright__ = "Copyright 2020, The Perception4all Project"
__credits__ = ["Afnan Balghaith", "Thariq Khalid"]

__maintainer__ = "Thariq Khalid"
__email__ = "thariq.khalid@gmail.com"
__status__ = "Research and Development"

"""

import torch
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# local imports
import data_loader
from networks import vanilla_cnn, VGG, resnet
from config import *
import cv2


# Tensorflow is static graph programming & PyTorch is Dynamic graph programming

def train(trainloader, valloader, experiment_name):
    writer_train = SummaryWriter('experiments/runs/{}/train'.format(experiment_name))
    writer_val = SummaryWriter('experiments/runs/{}/val'.format(experiment_name))

    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        training_loss=0.0
        val_loss=0.0

        # We will first run the train set iteration
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels=data  # 4, 8, 16, 32


            # zero the parameter gradients
            '''forward + backward + optimize
                refer to https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/ '''
            optimizer.zero_grad( )  # initialize all the gradients inside the network to 0
            outputs=net(inputs)
            loss=criterion(outputs, labels)  # error value
            loss.backward()
            optimizer.step()
            writer_train.add_graph(net, inputs)  # visualize model structure by using tensorboard

            # print statistics
            training_loss+=loss.item()

            if i % 2000 == 1999:  # print every 2000 mini-batches
                # ...log the running loss
                writer_train.add_scalar('Loss',training_loss / len(trainloader),epoch * len(trainloader) + i)  # Card 2
                print('[%d, %5d] training loss: %.3f' %(epoch + 1, i + 1, training_loss / len(trainloader)))
                training_loss = 0.0

        # Now we will run the validation set at the end of every epoch
        for j, val_data in enumerate(valloader, 0):
            with torch.no_grad():
                # get the inputs; data is a list of [inputs, labels]
                inputs_val, labels_val = val_data  # 4, 8, 16, 32
                # forward
                outputs_val = net(inputs_val)
                loss_val = criterion(outputs_val, labels_val)  # error value
                scheduler.step(loss_val)

                # print statistics
                val_loss += loss_val.item()

                if j % 2000 == 1999:
                    writer_val.add_scalar('Loss', val_loss / len(valloader), epoch * len(valloader) + j)  # Card 2
                    print('[%d, %5d] validation loss: %.3f' % (epoch + 1, j + 1, val_loss / len(valloader)))
                    val_loss = 0.0



    print("finished training phase")
    writer_train.close()
    writer_val.close()

# learning rate decay means start with 10e-6 and go to 10-3

def loss(net):
    criterion=nn.CrossEntropyLoss( )  # choice of your cost function, because you are doing multi class classification, CEL
    optimizer=optim.SGD(net.parameters( ), lr=LEARNING_RATE, momentum=0.9,weight_decay=5*(10)^4) # Adam, mAdagradSy with the optimizer
    return criterion, optimizer
def LRdecay(optimizer):
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.10, patience=10, verbose=False,)
    return scheduler



if __name__ == '__main__':
    # Card 3
    parser=ArgumentParser(description="Arguments for training the neural network according to the code design")
    parser.add_argument("--exp_name", type=str, help="the experiment name")

    args=parser.parse_args()
    experiment_name=args.exp_name

    trainloader, _, valloader=data_loader.datasetL()
    #net=vanilla_cnn.Net()
    #net=VGG.Net()
    net = resnet.ResNet()

    criterion, optimizer=loss(net)
    scheduler=LRdecay(optimizer)
    train(trainloader, valloader, experiment_name)

    PATH='experiments/models/VGG/A-architecture.pth'  # Card 3
    torch.save(net.state_dict( ), PATH)
