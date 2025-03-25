# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, stride = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size = 3, stride = stride, padding = 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(c_out)
        )
        
        self.identity = nn.Sequential()
        if stride > 1:
            self.identity = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size = 1, stride = stride, bias=False),
                nn.BatchNorm2d(c_out)
            )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.identity(x)      
        x = self.conv(x)
        x = x + identity
        return self.relu(x)

class CBAM(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(c, c // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c // reduction, c, 1, bias=False)
        )

        # Spatial Attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = torch.sigmoid(avg_out + max_out)
        x = x * ca

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * sa

        return x

class DecoderBlock(nn.Module):
    def __init__(self, c_in, c_out, use_cbam=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(c_out)

    def forward(self, x):
        x = self.conv(x)
        if self.use_cbam:
            return self.cbam(x)
        else:
            return x

class ResNet34_UNet(nn.Module):
    def __init__(self, c_in=3, c_out=2, use_cbam=True):
        super().__init__()
        
        # Encoder Resnet34
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.res1 = nn.Sequential(
            ResidualBlock(64,64,1),
            ResidualBlock(64,64,1),
            ResidualBlock(64,64,1)
        )
        
        self.res2 = nn.Sequential(
            ResidualBlock(64,128,2),
            ResidualBlock(128,128,1),
            ResidualBlock(128,128,1),
            ResidualBlock(128,128,1)
        )
        
        self.res3 = nn.Sequential(
            ResidualBlock(128,256,2),
            ResidualBlock(256,256,1),
            ResidualBlock(256,256,1),
            ResidualBlock(256,256,1),
            ResidualBlock(256,256,1),
            ResidualBlock(256,256,1)
        )
        
        self.res4 = nn.Sequential(
            ResidualBlock(256,512,2),
            ResidualBlock(512,512,1),
            ResidualBlock(512,512,1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size = 1, stride = 1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.center = DecoderBlock(768,32,use_cbam)
        
        self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(288,32,use_cbam)
        
        self.up2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(160,32,use_cbam)
        
        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(96,32,use_cbam)
        
        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(32,32,use_cbam)
        
        self.up5 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec5 = DecoderBlock(32,32,use_cbam)
        
        self.final_conv = nn.Conv2d(32, c_out, kernel_size=1)
        
    def forward(self, x):
        # Down
        x1 = self.conv1(x)
        x2 = self.maxPool(x1)
        
        x3 = self.res1(x2)
        x4 = self.res2(x3)
        x5 = self.res3(x4)
        x6 = self.res4(x5)
        
        # Bottleneck
        x7 = self.center(torch.cat([self.conv2(x6),x6], dim=1))

        # Up
        x8 = self.dec1(torch.cat([self.up1(x7),x5], dim=1))
        x9 = self.dec2(torch.cat([self.up2(x8),x4], dim=1))
        x10 = self.dec3(torch.cat([self.up3(x9),x3], dim=1))
        x11 = self.dec4(self.up4(x10))
        x12 = self.dec5(self.up5(x11))

        # Output
        return self.final_conv(x12)
        
        