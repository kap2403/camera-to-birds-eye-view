from collections import OrderedDict

import torch
import torch.nn as nn
from spt import BilinearInterpolation

from utils import device

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, init_features=16):
        super(UNet, self).__init__()
        theta = torch.tensor([[0.0, 0.8841865353311344, -253.37277367000263], [0.049056392233805146, 0.5285437237795494, -183.265385638118], [-0.0, 0.001750144780726984, -0.5285437237795492]])   
        self.theta = theta.to(device)
        features = init_features
        self.STN = BilinearInterpolation()  
        self.dropout = nn.Dropout(p=0.1)
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5 = UNet._block(features * 8, features * 16, name="enc5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        
        
        
        self.joiner1 = UNet._block(features,features,name = 'joiner1')
        self.joiner2 = UNet._block(features*2,features*2,name = 'joiner2')
        self.joiner3 = UNet._block(features*4,features*4,name = 'joiner3')
        self.joiner4 = UNet._block(features*8,features*8,name = 'joiner4')

        
        #self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        
        
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        drp1 = self.dropout(self.pool1(enc1))
        enc2 = self.encoder2(drp1)
        drp2 = self.dropout(self.pool2(enc2))
        enc3 = self.encoder3(drp2)
        drp3 = self.dropout(self.pool3(enc3))
        enc4 = self.encoder4(drp3)
        drp4 = self.dropout(self.pool4(enc4))
        enc5 = self.encoder5(drp4)
        
        
        #print(enc1.shape,enc2.shape,enc3.shape,enc4.shape,enc5.shape)
        
        spt1 = self.STN(enc1,self.theta)
        jon1 = self.joiner1(spt1)
        #print('1',jon1.shape)
        spt2 = self.STN(enc2,self.theta )
        jon2 = self.joiner2(spt2)
        #print('2',jon2.shape)
        spt3 = self.STN(enc3,self.theta )
        jon3 = self.joiner3(spt3)
        
        spt4 = self.STN(enc4,self.theta )
        jon4 = self.joiner4(spt4)
        
        #print(jon1.shape,jon2.shape,jon4.shape)
        #print(enc1.shape,enc2.shape,enc3.shape,enc4.shape,enc5.shape,)

        #bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, jon4), dim=1)
        drp4 = self.dropout(dec4)
        
        dec4 = self.decoder4(drp4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, jon3), dim=1)
        drp3 = self.dropout(dec3)
        
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, jon2), dim=1)
        drp2 = self.dropout(dec2)
        
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, jon1), dim=1)
        drp1 = self.dropout(dec1)
        
        dec1 = self.decoder1(drp1)
        
        return torch.softmax(self.conv(dec1), dim =1)
    
    

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )