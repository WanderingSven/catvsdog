import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNet(nn.Module):

    def __init__(self,debug=0):
        super().__init__()

        self.conv1_1 = nn.Conv2d(3,64,7,stride=2,padding=3)
        self.bn1_1 = nn.BatchNorm2d(64)
        

        self.conv2_1 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)

        self.conv4_1 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.conv4_2 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.bn4_2 = nn.BatchNorm2d(64)
        

        self.conv5_1 = nn.Conv2d(64,128,3,stride=2,padding=1)
        self.bn5_1 = nn.BatchNorm2d(128)
        self.conv5_2 = nn.Conv2d(128,128,3,stride=1,padding=1)
        self.bn5_2 = nn.BatchNorm2d(128)
        self.res_1 = nn.Conv2d(64,128,1,stride=2)
        self.bn_res_1 = nn.BatchNorm2d(128)

        self.conv6_1 = nn.Conv2d(128,128,3,stride=1,padding=1)
        self.bn6_1 = nn.BatchNorm2d(128)
        self.conv6_2 = nn.Conv2d(128,128,3,stride=1,padding=1)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.conv7_1 = nn.Conv2d(128,128,3,stride=1,padding=1)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128,128,3,stride=1,padding=1)
        self.bn7_2 = nn.BatchNorm2d(128)

        self.conv8_1 = nn.Conv2d(128,128,3,stride=1,padding=1)
        self.bn8_1 = nn.BatchNorm2d(128)
        self.conv8_2 = nn.Conv2d(128,128,3,stride=1,padding=1)
        self.bn8_2 = nn.BatchNorm2d(128)
        

        self.conv9_1 = nn.Conv2d(128,256,3,stride=2,padding=1)
        self.bn9_1 = nn.BatchNorm2d(256)
        self.conv9_2 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn9_2 = nn.BatchNorm2d(256)
        self.res_2 = nn.Conv2d(128,256,1,stride=2)
        self.res_bn_2 = nn.BatchNorm2d(256)

        self.conv10_1 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn10_1 = nn.BatchNorm2d(256)
        self.conv10_2 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn10_2 = nn.BatchNorm2d(256)

        self.conv11_1 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn11_1 = nn.BatchNorm2d(256)
        self.conv11_2 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn11_2 = nn.BatchNorm2d(256)

        self.conv12_1 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn12_1 = nn.BatchNorm2d(256)
        self.conv12_2 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn12_2 = nn.BatchNorm2d(256)

        self.conv13_1 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn13_1 = nn.BatchNorm2d(256)
        self.conv13_2 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn13_2 = nn.BatchNorm2d(256)

        self.conv14_1 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn14_1 = nn.BatchNorm2d(256)
        self.conv14_2 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.bn14_2 = nn.BatchNorm2d(256)
        

        self.conv15_1 = nn.Conv2d(256,512,3,stride=2,padding=1)
        self.bn15_1 = nn.BatchNorm2d(512)
        self.conv15_2 = nn.Conv2d(512,512,3,stride=1,padding=1)
        self.bn15_2 = nn.BatchNorm2d(512)
        self.res_3 = nn.Conv2d(256,512,1,stride=2)
        self.res_bn_3 = nn.BatchNorm2d(512)
        
        self.conv16_1 = nn.Conv2d(512,512,3,stride=1,padding=1)
        self.bn16_1 = nn.BatchNorm2d(512)
        self.conv16_2 = nn.Conv2d(512,512,3,stride=1,padding=1)
        self.bn16_2 = nn.BatchNorm2d(512)

        self.conv17_1 = nn.Conv2d(512,512,3,stride=1,padding=1)
        self.bn17_1 = nn.BatchNorm2d(512)
        self.conv17_2 = nn.Conv2d(512,512,3,stride=1,padding=1)
        self.bn17_2 = nn.BatchNorm2d(512)

        self.GAvg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512,2)


    def forward(self,x):
        # x [bacth,3,W,H] ---> [batch,C,W/2,H/2]
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        residual = x
        
        # x [bacth,C,W/2,H/2] ---> [batch,C,W/2,H/2]
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.bn2_2(self.conv2_2(x))
        x = F.relu(x + residual)
        residual = x
        
        # x [bacth,C,W/2,H/2] ---> [batch,C,W/2,H/2]
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = self.bn3_2(self.conv3_2(x))
        x = F.relu(x + residual)
        residual = x 

        # x [bacth,C,W/2,H/2] ---> [batch,C,W/2,H/2]
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = self.bn4_2(self.conv4_2(x))
        x = F.relu(x + residual)
        residual = x

        # x [bacth,C,W/2,H/2] ---> [batch,2C,W/4,H/4]
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = self.bn5_2(self.conv5_2(x))
        x = F.relu(x+self.bn_res_1(self.res_1(residual)))
        residual = x
        
        # x [bacth,2C,W/4,H/4] ---> [batch,2C,W/4,H/4]
        x = F.relu(self.bn6_1(self.conv6_1(x)))
        x = self.bn6_2(self.conv6_2(x))
        x = F.relu(x+residual)
        residual = x

        # x [bacth,2C,W/4,H/4] ---> [batch,2C,W/4,H/4]
        x = F.relu(self.bn7_1(self.conv7_1(x)))
        x = self.bn7_2(self.conv7_2(x))
        x = F.relu(x+residual)
        residual = x

        # x [bacth,2C,W/4,H/4] ---> [batch,2C,W/4,H/4]      
        x = F.relu(self.bn8_1(self.conv8_1(x)))
        x = self.bn8_2(self.conv7_2(x))
        x = F.relu(x+residual)
        residual = x

        # x [bacth,2C,W/4,H/4] ---> [batch,4C,W/8,H/8] 
        x = F.relu(self.bn9_1(self.conv9_1(x)))
        x = self.bn9_2(self.conv9_2(x))
        x = F.relu(x+self.res_bn_2(self.res_2(residual)))
        residual = x

        # x [bacth,4C,W/8,H/8] ---> [batch,4C,W/8,H/8] 
        x = F.relu(self.bn10_1(self.conv10_1(x)))
        x = self.bn10_2(self.conv10_2(x))
        x = F.relu(x+residual)
        residual = x

        # x [bacth,4C,W/8,H/8] ---> [batch,4C,W/8,H/8] 
        x = F.relu(self.bn11_1(self.conv11_1(x)))
        x = self.bn11_2(self.conv11_2(x))
        x = F.relu(x+residual)
        residual = x

        # x [bacth,4C,W/8,H/8] ---> [batch,4C,W/8,H/8] 
        x = F.relu(self.bn12_1(self.conv12_1(x)))
        x = self.bn12_2(self.conv12_2(x))
        x = F.relu(x+residual)
        residual = x

        # x [bacth,4C,W/8,H/8] ---> [batch,4C,W/8,H/8] 
        x = F.relu(self.bn13_1(self.conv13_1(x)))
        x = self.bn13_2(self.conv13_2(x))
        x = F.relu(x+residual)
        residual = x

        # x [bacth,4C,W/8,H/8] ---> [batch,4C,W/8,H/8] 
        x = F.relu(self.bn14_1(self.conv14_1(x)))
        x = self.bn14_2(self.conv14_2(x))
        x = F.relu(x+residual)
        residual = x

        # x [bacth,4C,W/8,H/8] ---> [batch,8C,W/16,H/16]
        x = F.relu(self.bn15_1(self.conv15_1(x)))
        x = self.bn15_2(self.conv15_2(x))
        x = F.relu(x+self.res_bn_3(self.res_3(residual)))
        residual = x

        # x [bacth,8C,W/16,H/16] ---> [batch,8C,W/16,H/16]
        x = F.relu(self.bn16_1(self.conv16_1(x)))
        x = self.bn16_2(self.conv16_2(x))
        x = F.relu(x+residual)
        residual = x

        # x [bacth,8C,W/16,H/16] ---> [batch,8C,W/16,H/16]
        x = F.relu(self.bn17_1(self.conv17_1(x)))
        x = self.bn17_2(self.conv17_2(x))
        x = F.relu(x+residual)
        
        # x[batch,8C,W/16,H/16] --->[ batch,8C,1,1]
        x = self.GAvg_pool(x)

        #[ batch,8C,1,1] ---> [batch,8C]
        x = x.view(x.size(0),x.size(1))
        
        #[batch,8C]---->[batch,2]
        return self.fc(x)








