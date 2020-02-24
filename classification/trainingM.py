"""
__author__ = "Afnan Qalas Balghaith, Thariq Khalid"
__copyright__ = "Copyright 2020, The Perception4all Project"
__credits__ = ["Afnan Balghaith", "Thariq Khalid"]

__maintainer__ = "Thariq Khalid"
__email__ = "thariq.khalid@gmail.com"
__status__ = "Research and Development"

"""

# library imports
import torch
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import matplotlib.pyplot as plt

# local imports
import data_loader
from networks import CNN


def train(trainloader, experiment_name):
    writer = SummaryWriter('experiments/runs/{}'.format(experiment_name))

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_graph(net, inputs)

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                # ...log the running loss
                writer.add_scalar('training loss',
                                  running_loss / 1000,
                                  epoch * len(trainloader) + i) # Card 2

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print("finished training phase")
    writer.close()


if __name__ == '__main__':
    # Card 3
    parser = ArgumentParser(description="Arguments for training the neural network according to the code design")
    parser.add_argument('--exp_name', type=str, help="the experiment name")

    args = parser.parse_args()
    experiment_name = args.exp_name

    trainloader, _, _ = data_loader.datasetL()
    net = CNN.Net()


    criterion, optimizer = net.loss()
    train(trainloader, experiment_name)

    PATH = 'experiments/models/cifar_net.pth' # Card 3
    torch.save(net.state_dict(), PATH)



