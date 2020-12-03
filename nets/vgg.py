import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Vgg(nn.Module):

    def __init__(self,debug=0):
        super(Vgg,self).__init__()
        self.debug = debug

        self.conv_1_1 = nn.Conv2d(3,64,3,stride=1,padding=1)
        self.bn_1_1 = nn.BatchNorm2d(64)
        self.conv_1_2 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.bn_1_2 = nn.BatchNorm2d(64)
        self.max_pool_1 = nn.MaxPool2d(2,stride=2)

        self.conv_2_1 = nn.Conv2d(64,128,3,stride=1,padding=1)
        self.bn_2_1 = nn.BatchNorm2d(128)
        self.conv_2_2 = nn.Conv2d(128,128,3,stride=1,padding=1)
        self.bn_2_2 = nn.BatchNorm2d(128)
        self.max_pool_2 = nn.MaxPool2d(2,stride=2)

        self.conv_3_1 = nn.Conv2d(128,256,3,stride=1,padding=1)
        self.bn_3_1 = nn.BatchNorm2d(256)
        self.conv_3_2 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn_3_2 = nn.BatchNorm2d(256)
        self.conv_3_3 = nn.Conv2d(256,256,1,stride=1)
        self.bn_3_3 = nn.BatchNorm2d(256)
        self.max_pool_3 = nn.MaxPool2d(2,stride=2)

        self.conv_4_1 = nn.Conv2d(256,512,3,stride=1,padding=1)
        self.bn_4_1 = nn.BatchNorm2d(512)
        self.conv_4_2 = nn.Conv2d(512,512,3,stride=1,padding=1)
        self.bn_4_2 = nn.BatchNorm2d(512)
        self.conv_4_3 = nn.Conv2d(512,512,1,stride=1)
        self.bn_4_3 = nn.BatchNorm2d(512)
        self.max_pool_4 = nn.MaxPool2d(2,stride=2)

        self.conv_5_1 = nn.Conv2d(512,512,3,stride=1,padding=1)
        self.bn_5_1 = nn.BatchNorm2d(512)
        self.conv_5_2 = nn.Conv2d(512,512,3,stride=1,padding=1)
        self.bn_5_2 = nn.BatchNorm2d(512)
        self.conv_5_3 = nn.Conv2d(512,512,1,stride=1)
        self.bn_5_3 = nn.BatchNorm2d(512)
        self.max_pool_5 = nn.MaxPool2d(2,stride=2)

        self.fc_1 = nn.Linear(8192,4096)
        self.fc_2 = nn.Linear(4096,4096)
        self.fc_3 = nn.Linear(4096,2)

        self.Dropout1 = nn.Dropout(0.5)
        self.Dropout2 = nn.Dropout(0.5)

        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self,x):
        
        #print(x.size())
        x = F.relu(self.bn_1_1(self.conv_1_1(x)))
        x = F.relu(self.bn_1_2(self.conv_1_2(x)))
        x = self.max_pool_1(x)
        #print(x.size())

        x = F.relu(self.bn_2_1(self.conv_2_1(x)))
        x = F.relu(self.bn_2_2(self.conv_2_2(x)))
        x = self.max_pool_2(x)
        #print(x.size())

        x = F.relu(self.bn_3_1(self.conv_3_1(x)))
        x = F.relu(self.bn_3_2(self.conv_3_2(x)))
        x = F.relu(self.bn_3_3(self.conv_3_3(x)))
        x = self.max_pool_3(x)
        #print(x.size())

        x = F.relu(self.bn_4_1(self.conv_4_1(x)))
        x = F.relu(self.bn_4_2(self.conv_4_2(x)))
        x = F.relu(self.bn_4_3(self.conv_4_3(x)))
        x = self.max_pool_4(x)
        #print(x.size())

        x = F.relu(self.bn_5_1(self.conv_5_1(x)))
        x = F.relu(self.bn_5_2(self.conv_5_2(x)))
        x = F.relu(self.bn_5_3(self.conv_5_3(x)))
        x = self.max_pool_5(x)
        #print(x.size())

        x = x.contiguous().view(x.size(0),-1)
        x = self.Dropout1(F.relu(self.fc_1(x)))
        x = self.Dropout2(F.relu(self.fc_2(x)))
        x = self.fc_3(x)

        return x 

class SE_bolck(nn.Module):
    
    def __init__(self,channel,R):
        super().__init__()

        self.linear1 = nn.Linear(channel,channles//R)
        self.linear2 = nn.Linear(channel//R,channels)
    
    def forward(self,x):
        """
            x : (batch,channel,width,height)
        """

        x = x.sum(-1,-2) / (x.size(-1)*x.size(-2)) #squeeze operate

        x = F.relu(self.linear1(x))                   
        x = nn.Sigmoid(self.linear2(x))

        return x.contiguous().view(x.size(0),x.size(1),1,1)


