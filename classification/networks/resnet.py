"""
__author__ = "Afnan Qalas Balghaith, Thariq Khalid"
__copyright__ = "Copyright 2020, The Perception4all Project"
__credits__ = ["Afnan Balghaith", "Thariq Khalid"]

__maintainer__ = "Thariq Khalid"
__email__ = "thariq.khalid@gmail.com"
__status__ = "Research and Development"

"""

import torch.nn.functional as F
from torch import nn
import torch

conv2_x = [64,64,64,256]
conv2_x_identity = [256,64,64,256]
conv3_x = [256,128,128,512]
conv3_x_identity = [512,128,128,512]
conv4_x = [512,256,256,1024]
conv4_x_identity = [1024,256,256,1024]
conv5_x = [1024,512,512,2048]
conv5_x_identity = [2048,512,512,2048]
s = 2


class identityblock_module(nn.Module):
    def __init__ (self,identity_block):
        super(identityblock_module,self).__init__()

        # first component of main path
        self.conv1 = nn.Conv2d(in_channels = identity_block[0],out_channels = identity_block[1],
                               kernel_size = (1,1),stride = (1,1))
        self.BN1 = nn.BatchNorm2d(num_features = identity_block[1])
        # second component of main path
        self.conv2 = nn.Conv2d(in_channels = identity_block[1],out_channels = identity_block[2],
                               kernel_size = (3,3),stride = (1,1),padding = (1,1))
        self.BN2 = nn.BatchNorm2d(num_features = identity_block[2])
        # third component of main path
        self.conv3 = nn.Conv2d(in_channels = identity_block[2],out_channels = identity_block[3],
                               kernel_size = (1,1),stride = (1,1))
        self.BN3 = nn.BatchNorm2d(num_features = identity_block[3])

    def forward (self,x):
        x_shortcut = x
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.relu(self.BN2(self.conv2(x)))
        x = self.BN3(self.conv3(x))
        x = F.relu(torch.add(x_shortcut,x))
        return x


class convblock_module(nn.Module):
    def __init__ (self,conv_block,s):
        super(convblock_module,self).__init__()
        # first component of main path
        self.conv1 = nn.Conv2d(in_channels = conv_block[0],out_channels = conv_block[1],
                               kernel_size = (1,1),stride = (s,s))
        self.BN1 = nn.BatchNorm2d(num_features = conv_block[1])
        # second component of main path
        self.conv2 = nn.Conv2d(in_channels = conv_block[1],out_channels = conv_block[2],
                               kernel_size = (3,3),stride = (1,1),padding = (1,1))
        self.BN2 = nn.BatchNorm2d(num_features = conv_block[2])
        # third component of main path
        self.conv3 = nn.Conv2d(in_channels = conv_block[2],out_channels = conv_block[3],
                               kernel_size = (1,1),stride = (1,1))
        self.BN3 = nn.BatchNorm2d(num_features = conv_block[3])

        # component of shortcut path
        self.conv3_Shortcut = nn.Conv2d(in_channels = conv_block[0],out_channels = conv_block[3],
                                        kernel_size = (1,1),stride = (s,s))
        self.BN_Shortcut = nn.BatchNorm2d(num_features = conv_block[3])

    def forward (self,x):
        x_shortcut = x
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.relu(self.BN2(self.conv2(x)))
        x = self.BN3(self.conv3(x))
        x_shortcut = self.BN_Shortcut(self.conv3_Shortcut(x_shortcut))
        x = F.relu(torch.add(x_shortcut,x))
        return x


class ResNet(nn.Module):
    def __init__ (self):
        super(ResNet,self).__init__()

        # first stage
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 64,
                               kernel_size = (7,7),stride = (2,2),padding = (3,3))
        self.BN = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size = (3,3),stride = (2,2),ceil_mode = True)

        # second stage
        self.conv2_1 = convblock_module(conv2_x,1)
        self.conv2_2 = identityblock_module(conv2_x_identity)
        self.conv2_3 = identityblock_module(conv2_x_identity)

        # third stage
        self.conv3_1 = convblock_module(conv3_x,2)
        self.conv3_2 = identityblock_module(conv3_x_identity)
        self.conv3_3 = identityblock_module(conv3_x_identity)
        self.conv3_4 = identityblock_module(conv3_x_identity)

        # forth stage
        self.conv4_1 = convblock_module(conv4_x,2)
        self.conv4_2 = identityblock_module(conv4_x_identity)
        self.conv4_3 = identityblock_module(conv4_x_identity)
        self.conv4_4 = identityblock_module(conv4_x_identity)
        self.conv4_5 = identityblock_module(conv4_x_identity)
        self.conv4_6 = identityblock_module(conv4_x_identity)

        # fifth stage
        self.conv5_1 = convblock_module(conv5_x,2)
        self.conv5_2 = identityblock_module(conv5_x_identity)
        self.conv5_3 = identityblock_module(conv5_x_identity)

        self.avgPool = nn.AvgPool2d(kernel_size = (2,2),stride = (1,1))
        self.fc = nn.Linear(in_features = 2048 * 6 * 6,out_features = 1000)

    def forward (self,x):
        # stage 1
        x = self.BN(self.conv1(x))
        x = self.pool(F.relu(x))

        # stage 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        # stage 3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        # stage 4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        # stage 5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        # final stage
        x = self.avgPool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
