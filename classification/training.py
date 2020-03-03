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
from networks import vanilla_cnn
from config import *


# Tensorflow is static graph programming & PyTorch is Dynamic graph programming

def train(trainloader, valloader, experiment_name):
    writertrain=SummaryWriter('experiments/runs/train')
    writerval=SummaryWriter('experiments/runs/val')


    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss=0.0
        val_loss=0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels=data  # 4, 8, 16, 32

            # zero the parameter gradients
            optimizer.zero_grad( )  # initialize all the gradients inside the network to 0

            # forward + backward + optimize
            outputs=net(inputs)
            loss=criterion(outputs, labels)  # error value
            loss.backward( )
            optimizer.step( )
            writertrain.add_graph(net, inputs)  # visualize model structure by using tensorboard

            # print statistics
            running_loss+=loss.item( )

            if i % 2000 == 1999:  # print every 2000 mini-batches
                # ...log the running loss
                writertrain.add_scalar('training loss',running_loss / len(trainloader),epoch * len(trainloader) + i)  # Card 2

                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / len(valloader)))
                running_loss=0.0
        else:
            with torch.no_grad( ):
                for j, valdata in enumerate(valloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputsval, labelsval=valdata  # 4, 8, 16, 32

                    # forward
                    outputsval = net(inputsval)
                    lossval = criterion(outputsval, labelsval)  # error value

                    # print statistics
                    val_loss += lossval.item()

                    if j % 100 == 99:  # print every 2000 mini-batches
                        # ...log the running loss
                        writerval.add_scalar('validation loss', val_loss / len(valloader), epoch * len(valloader) + j)  # Card 2
                        print('[%d, %5d] validation loss: %.3f' % (epoch + 1, j + 1, val_loss / len(valloader)))
                        val_loss = 0.0


    print("finished training phase")
    writer.close()


# learning rate decay means start with 10e-6 and go to 10-3

def loss(net):
    criterion=nn.CrossEntropyLoss( )  # choice of your cost function, because you are doing multi class classification, CEL
    optimizer=optim.SGD(net.parameters( ), lr=LEARNING_RATE, momentum=0.9)  # Adam, mAdagradSy with the optimizer
    return criterion, optimizer


if __name__ == '__main__':
    # Card 3
    parser=ArgumentParser(description="Arguments for training the neural network according to the code design")
    parser.add_argument("--exp_name", type=str, help="the experiment name")

    args=parser.parse_args( )
    experiment_name=args.exp_name

    trainloader, _, valloader=data_loader.datasetL( )
    net=vanilla_cnn.Net( )

    criterion, optimizer=loss(net)
    train(trainloader, valloader, experiment_name)

    PATH='experiments/models/cifar_net.pth'  # Card 3
    torch.save(net.state_dict( ), PATH)
