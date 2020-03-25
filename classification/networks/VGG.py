"""
__author__ = "Afnan Qalas Balghaith, Thariq Khalid"
__copyright__ = "Copyright 2020, The Perception4all Project"
__credits__ = ["Afnan Balghaith", "Thariq Khalid"]

__maintainer__ = "Thariq Khalid"
__email__ = "thariq.khalid@gmail.com"
__status__ = "Research and Development"

"""

import torch.nn as nn
import torch.nn.functional as F

'''
Important notes, should be highlighted!
- I have to create the 6 architectures of VGG A, A-LRN, B, C, D,E,
 the differences in the number of conv layers. 
- kernel size = 3x3, stride = 1, padding =1 for conv layers, kernel size for max pooling = 2x2 with stride = 2.
- 3 fully connected layers, the first two one have 4096 channels, and the last one has 1000 channels,
 but for our dataset(CIFAR-10) it should be 10 channels
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(2,2), padding=(1,1))
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(2,2), padding=(1,1))
        self.pool=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv3=nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(3,3),stride=(2,2), padding=(1,1))
        self.conv4=nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3),stride=(2,2), padding=(1,1))
        self.conv5=nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3),stride=(2,2), padding=(1,1))
        self.conv6=nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(3,3),stride=(2,2), padding=(1,1))
        self.fc1=nn.Linear(in_features=512 * 3 * 3,out_features=4096)
        self.fc2=nn.Linear(in_features=4096,out_features=4096)
        self.fc3=nn.Linear(in_features=4096,out_features=10)
        self.dropout=nn.Dropout(0.5)

    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=F.relu(self.conv3(x))
        x=self.pool(F.relu(self.conv4(x)))
        x=F.relu(self.conv5(x))
        x=self.pool(F.relu(self.conv6(x)))
        x=F.relu(self.conv6(x))
        x=self.pool(F.relu(self.conv6(x)))
        x=x.view(-1, 512 * 3 * 3)
        x=F.relu(self.dropout(self.fc1(x)))
        x=F.relu(self.fc2(x))
        x=F.self.fc3(x)
        return x


        

