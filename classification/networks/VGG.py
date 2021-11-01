"""
__author__ = "Afnan Qalas, Thariq Khalid"
__copyright__ = "Copyright 2020, The Perception4all Project"
__credits__ = ["Afnan Qalas", "Thariq Khalid"]

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
- 3 fully connected layers, the first two ones have 4096 channels, and the last one has 1000 channels,
 but for our dataset(CIFAR-10) it should be 10 channels
'''

class VggNet(nn.Module):
    def __init__(self):
        super(VggNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 512, 3)
        self.fc1 = nn.Linear(512 * 3 * 3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = self.pool(F.relu((self.conv4(x))))
        x = F.relu((self.conv5(x)))
        x = self.pool(F.relu((self.conv6(x))))
        x = F.relu((self.conv6(x)))
        x = self.pool(F.relu((self.conv6(x))))
        x = x.view( x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

