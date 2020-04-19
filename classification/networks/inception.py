import torch.nn.functional as F
from torch import nn
import torch

# we'll put the input to the inception block as the first value of the list
inception_3a = [192, 64, 96, 128, 16, 32, 32]
inception_3b = [256, 128, 128, 192, 32, 96, 64]
inception_4a = [480, 192, 96, 208, 16, 48, 64]
inception_4b = [512, 160, 112, 224, 24, 64, 64]
inception_4c = [512, 128, 128, 256, 24, 64, 64]
inception_4d = [512, 112, 144, 288, 32, 64, 64]
inception_4e = [528, 256, 160, 320, 32, 128, 128]
inception_5a = [832, 256, 160, 320, 32, 128, 128]
inception_5b = [832, 384, 192, 384, 48, 128, 128]

class inception_module():
    def __init__(self, inception_block):
        self.conv1 = nn.Conv2d(inception_block[0], inception_block[1], 1)
        self.conv3_r = nn.Conv2d(inception_block[0], inception_block[2], 1)
        self.conv3 = nn.Conv2d(inception_block[2], inception_block[3], 3)
        self.conv5_r = nn.Conv2d(inception_block[0], inception_block[4], 1)
        self.conv5 = nn.Conv2d(inception_block[4], inception_block[5], 5)
        self.pool = nn.MaxPool2d(3,3)
        self.conv1_m = nn.Conv2d(inception_block[5], inception_block[5], 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.relu(self.conv3(F.relu(self.conv3_r(x))))
        x3 = F.relu(self.conv5(F.relu(self.conv5_r(x))))
        x4 = F.relu(self.conv1_m(self.pool(x)))
        x_final = torch.cat(x1, x2, x3, x4)
        return x_final