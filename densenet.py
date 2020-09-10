import torch
import os
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import math

#from torchviz import make_dot
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
class SingleLayer(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(inchannel)
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=5,padding=2)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out




class Transition(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inchannel)
        self.conv1 = nn.Conv2d(inchannel,outchannel, kernel_size=1)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        return out

class SDB( nn.Module):
    def __init__(self):
        super(SDB, self).__init__()
        self.bn=nn.BatchNorm2d(64)
        self.conv=nn.Conv2d(64, 32, 5, padding=2)
        self.dense1=self._make_dense()
        self.dense2 = self._make_dense()
        self.trans1=Transition(192, 32)
        self.trans2 = Transition(192, 32)

    def _make_dense(self):
        layers = []
        layers.append(SingleLayer(32,32))
        layers.append(SingleLayer(64, 32))
        layers.append(SingleLayer(96, 32))
        layers.append(SingleLayer(128, 32))
        return nn.Sequential(*layers)

    def forward(self,x,y):
        s = F.relu(self.bn(torch.cat([x, y], 1)))
        s = self.conv(s)
        s1 = self.trans1(torch.cat([x,self.dense1(s)],1))
        s2 = self.trans2(torch.cat([y,self.dense2(s)],1))
        return s1,s2

class IFB( nn.Module):
    def __init__(self):
        super(IFB, self).__init__()
        self.dense = self._make_dense_2()
        self.trans = Transition(192, 64)
    def _make_dense_2(self):
        layers = []
        layers.append(SingleLayer(64, 32))
        layers.append(SingleLayer(96, 32))
        layers.append(SingleLayer(128, 32))
        layers.append(SingleLayer(160, 32))
        return nn.Sequential(*layers)
    def forward(self,x):
        out=self.trans(self.dense(x))
        return out





class SDNet(nn.Module):
    def __init__(self):
        super(SDNet, self).__init__()

        self.conv1_0= nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_1 =nn.Conv2d(3, 32, 3, padding=1)
        self.SDB0 = SDB()
        self.SDB1 = SDB()
        self.SDB2 = SDB()
        self.SDB3 = SDB()
        self.SDB4 = SDB()
        self.SDB5 = SDB()
        self.SDB6 = SDB()

        self.conv3_1=nn.Conv2d(256, 32, 5, padding=2)
        self.conv3_2 = nn.Conv2d(256, 32, 5, padding=2)
        self.conv4_1 = nn.Conv2d(32, 3, 5, padding=2)
        self.conv4_2 = nn.Conv2d( 32, 3,5, padding=2)
        self.bn1_1=nn.BatchNorm2d(32)
        self.bn1_2= nn.BatchNorm2d(32)
        self.bn2_1=nn.BatchNorm2d(256)
        self.bn2_2 = nn.BatchNorm2d(256)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()
    def forward(self, x,y):
        s1_0_0 = self.conv1_0(x)
        s2_0_0 = self.conv1_1(y)

        s1_0,s2_0=self.SDB0(s1_0_0,s2_0_0)

        s1_1, s2_1 = self.SDB1(s1_0_0, s2_0_0)
        s1_1_0= F.avg_pool2d(s1_1 ,2)
        s2_1_0 = F.avg_pool2d(s2_1, 2)

        s1_2, s2_2 = self.SDB2(s1_1_0, s2_1_0)
        s1_2_0=F.interpolate(s1_2,(224,320),mode='bilinear')
        s2_2_0 = F.interpolate(s2_2, (224, 320), mode='bilinear')

        s1_3, s2_3 = self.SDB3(s1_2, s2_2)
        s1_3_0 = F.interpolate(s1_3, (224, 320), mode='bilinear')
        s2_3_0 = F.interpolate(s2_3, (224, 320), mode='bilinear')

        s1_4, s2_4 = self.SDB4(s1_3, s2_3)
        s1_4 =F.interpolate(s1_4,(224,320),mode='bilinear')
        s2_4 =F.interpolate(s2_4,(224,320),mode='bilinear')
        s1_5, s2_5 = self.SDB5(s1_4, s2_4)
        s1_6, s2_6 = self.SDB6(s1_5, s2_5)
        s1=torch.cat([s1_0_0, s1_0, s1_1, s1_2_0, s1_3_0, s1_4, s1_5, s1_6], 1)
        s2=torch.cat([s2_0_0, s2_0, s2_1, s2_2_0, s2_3_0, s2_4, s2_5, s2_6], 1)
        s1=self.conv4_1(self.bn1_1(self.conv3_1(self.bn2_1(s1))+s1_0_0))
        s2=self.conv4_2(self.bn1_2(self.conv3_2(self.bn2_2(s2))+s2_0_0))
        return s1,s2

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.IFB0 = IFB()
        self.IFB1 = IFB()
        self.IFB2 = IFB()
        self.conv1=nn.Conv2d(6,64,7,padding=3)
        self.conv2 = nn.Conv2d(256, 64, 7, padding=3)
        self.bn1=nn.BatchNorm2d(256)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d( 64, 3,7, padding=3)
    def forward(self, a,b):
        x=torch.cat([a,b],1)
        x_0=self.conv1(x)
        x_1=self.IFB0(x_0)
        x_2=self.IFB1(x_1)
        x_3=self.IFB2(x_2)
        x_4= torch.cat([x_0,x_1,x_2,x_3],1)
        x=self.conv2(F.relu(self.bn1(x_4)))
        output=self.conv3(self.bn2(x))
        return output








