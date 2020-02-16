from networks import CNN
import torch
import data_loader
from torch.utils.tensorboard import SummaryWriter

def testing():

    _,testloader,classes = data_loader.datasetL()
    dataiter = iter(testloader)
    net = CNN.Net()
    images, labels = dataiter.next()
    # print images
    net.load_state_dict(torch.load('cifar_net.pth'))
    correct = 0
    total = 0

    # The code marked as 16022019 is for the section 6 in https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#assessing-trained-models-with-tensorboard
    # 1. gets the probability predictions in a test_size x num_classes Tensor
    # 2. gets the preds in a test_size Tensor
    # takes ~10 seconds to run
    class_probs = [] # 16022019
    class_preds = [] # 16022019

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output] # 16022019
            _, predicted = torch.max(outputs.data, 1)
            _, class_preds_batch = torch.max(output, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            class_probs.append(class_probs_batch) # 16022019
            class_preds.append(class_preds_batch) # 16022019

    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs]) # 16022019
    test_preds = torch.cat(class_preds) # 16022019

    for i in range(10): # 16022019
        add_pr_curve_tensorboard(i, test_probs, test_preds) # 16022019

# 16022019 helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]
    writer = SummaryWriter('runs/cnn_experiments_1')

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()


if __name__ == '__main__':
    testing()

    print("Finished Testing")
