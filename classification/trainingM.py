import data_loader
from networks import CNN
import torch

from torch.utils.tensorboard import SummaryWriter


def train(trainloader):
    writer = SummaryWriter('runs/cnn_experiments_1')

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
                                  epoch * len(trainloader) + i) # 16022019

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("finished training phase")
    writer.close()

if __name__ == '__main__':
    trainloader, _, _ = data_loader.datasetL()
    net = CNN.net
    criterion, optimizer = net.loss()
    train(trainloader)

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)



