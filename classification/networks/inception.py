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

#Aux blocks
inception_4b_aux = [512,512]
inception_4e_aux = [528,832]


class inception_module(nn.Module):
    def __init__(self, inception_block):
        super(inception_module , self).__init__( )
        self.conv1 = nn.Conv2d(in_channels=inception_block[0], out_channels=inception_block[1], kernel_size=(1,1), stride=(1,1))
        self.conv3_r = nn.Conv2d(in_channels=inception_block[0], out_channels=inception_block[2], kernel_size=(1,1), stride=(1,1),padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=inception_block[2],out_channels= inception_block[3], kernel_size=(3,3),stride=(1,1))
        self.conv5_r = nn.Conv2d(in_channels=inception_block[0], out_channels=inception_block[4],kernel_size=(1,1), stride=(1,1),padding=(1,1))
        self.conv5 = nn.Conv2d(in_channels=inception_block[4], out_channels=inception_block[5],kernel_size=(5,5), stride=(1,1),padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1),ceil_mode=True)
        self.conv1_m = nn.Conv2d(in_channels=inception_block[0], out_channels=inception_block[6], kernel_size=(1,1), stride=(1,1),padding=(1,1))

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv3(F.relu(self.conv3_r(x))))
        x3 = F.relu(self.conv5(F.relu(self.conv5_r(x))))
        x4 = F.relu(self.conv1_m(self.pool(x)))
        x_final = torch.cat((x1, x2, x3, x4),dim=1)
        return x_final

"""
• An average pooling layer with 5×5 filter size and stride 3, resulting in an 4×4×512 output
for the (4a), and 4×4×528 for the (4d) stage.
• A 1×1 convolution with 128 filters for dimension reduction and rectified linear activation. # I didn't understand why they say 128 filters while it's 1 1x1 conv 
• A fully connected layer with 1024 units and rectified linear activation.  
• A dropout layer with 70% ratio of dropped outputs.
• A linear layer with softmax loss as the classifier (predicting the same 1000 classes as the main classifier, but removed at inference time).
"""
class inceptionAux_module(nn.Module):
    def __init__ ( self , inceptionAux_block ) :
        super(inceptionAux_module , self).__init__( )
        self.conv = nn.Conv2d(in_channels=inceptionAux_block[0],out_channels=inceptionAux_block[1], kernel_size=(1,1), stride=(1,1))
        self.out_channels = inceptionAux_block[1]
        self.avgPool = nn.AvgPool2d(kernel_size = (5,5), stride= (3,3), ceil_mode=True)
        self.fc1 = nn.Linear(in_features = inceptionAux_block[1] , out_features=1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features=10)
        self.dropout = nn.Dropout(0.70)

    def forward( self, x):
        x = self.avgPool(x)
        x = F.relu(self.conv(x))
        x = x.view(-1, self.out_channels )
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.softmax(self.fc2(x), dim=0)
        return x


class InceptionNet(nn.Module):
    def __init__ ( self ) :
        super(InceptionNet, self).__init__( )
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        self.pool  = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2), ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=64 , out_channels=64, kernel_size=(1,1),stride=(1,1)) # padding = 0
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.module_3a = inception_module(inception_3a)
        self.module_3b = inception_module(inception_3b)
        self.module_4a = inception_module(inception_4a)
        self.module_4b = inception_module(inception_4b)
        self.module_4b_aux = inceptionAux_module(inception_4b_aux)
        self.module_4c = inception_module(inception_4c)
        self.module_4d=inception_module(inception_4d)
        self.module_4e=inception_module(inception_4e)
        self.module_4e_aux=inceptionAux_module(inception_4e_aux)
        self.module_5a = inception_module(inception_5a)
        self.module_5b = inception_module(inception_5b)
        self.avgPool = nn.AvgPool2d(kernel_size=(7,7), stride=(2,2), ceil_mode=True)
        self.dropout = nn.Dropout(0.40)
        # as the original work they put out_features = 1000 because of the dataset that is used, but for us the dataset that we will use has 10 class
        self.fc = nn.Linear(in_features = 1024 * 1 * 1, out_features= 10)

    def forward (self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv3(F.relu(self.conv2(x)))))
        x = self.module_3a(x)
        x = self.module_3b(x)
        x = self.pool(x)
        x = self.module_4a(x)
        x1 = x
        x = self.module_4b(x)
        x1 = self.module_4b_aux(x1)
        x = self.module_4c(x)
        x = self.module_4d(x)
        x2 = x
        print(x2.shape)
        x = self.module_4e(x)
        print(x.shape)
        x2 = self.module_4e_aux(x2)
        print(x2.shape)
        x = self.pool(x)
        x = self.module_5a(x)
        x = self.module_5b(x)
        x = self.avgPool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.softmax(self.fc(x), dim=0)
        return x,x1,x2











