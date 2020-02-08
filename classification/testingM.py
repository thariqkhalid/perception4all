# from classification import data_loader,CNN
import CNN
import torch
import data_loader

def testing():

    _,testloader,classes = data_loader.datasetL()
    dataiter = iter(testloader)
    net = CNN.Net()
    images, labels = dataiter.next()
    # print images
    net.load_state_dict(torch.load('cifar_net.pth'))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

if __name__ == '__main__':
    testing()
    print("Finished Testing")
