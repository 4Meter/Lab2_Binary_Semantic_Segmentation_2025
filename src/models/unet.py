# Implement your UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), #same convolution
        nn.ReLU()
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1), #same convolution1
        nn.ReLU()
    )
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    return x

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=2, device="cuda"):
        super().__init__()
        self.device = device
        
        self.conv1 = DoubleConv(c_in,64)       # (3,256,256) -> (64,256,256)
        self.down1 = nn.MaxPool2d(2)        # (64,256,256) -> (64,128,128)
        
        self.conv2 = DoubleConv(64,128)     # (64,128,128) -> (128,128,128)
        self.down2 = nn.MaxPool2d(2)        # (128,128,128) -> (128,64,64)
        
        self.conv3 = DoubleConv(128,256)    # (128,64,64) -> (256,64,64)
        self.down3 = nn.MaxPool2d(2)        # (256,64,64) -> (256,32,32)
        
        self.conv4 = DoubleConv(256,512)    # (256,32,32) -> (512,32,32)
        self.down4 = nn.MaxPool2d(2)        # (512,32,32) -> (512,16,16)
        
        self.conv5 = DoubleConv(512,1024)    # (512,16,16) -> (1024,16,16)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        
        self.conv6 = DoubleConv(1024,512)    
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        
        self.conv7 = DoubleConv(512,256)    
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        
        self.conv8 = DoubleConv(256,128)    
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.conv9 = DoubleConv(128,64)
        
        self.final_conv = nn.Conv2d(64, c_out, kernel_size=1)
        
    def forward(self, x):
        # Down
        x1 = self.conv1(x)
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))
        x4 = self.conv4(self.down3(x3))
        # Bottleneck
        x5 = self.conv5(self.down4(x4))
        # Up
        x6 = self.conv6(torch.cat([x4, self.up1(x5)], dim=1))
        x7 = self.conv7(torch.cat([x3, self.up2(x6)], dim=1))
        x8 = self.conv8(torch.cat([x2, self.up3(x7)], dim=1))
        x9 = self.conv9(torch.cat([x1, self.up4(x8)], dim=1))
        # Output
        return self.final_conv(x9)