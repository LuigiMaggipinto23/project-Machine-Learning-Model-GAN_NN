import torch
from torch import nn
from constants import *

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return nn.LeakyReLU(0.2, inplace=True)(x + self.conv_block(x))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input size: 128x128x3
            nn.Conv2d(3, FEATURES_D, 4, 2, 1), 	# 128x128x3 -> 64x64x32
            nn.LeakyReLU(0.2, inplace=True), 	# Leaky ReLU
            
            
            nn.Conv2d(FEATURES_D, FEATURES_D * 2, 4, 2, 1), # 64x64x32 -> 32x32x64
            nn.LeakyReLU(0.2, inplace=True),				# Leaky ReLU
            
            nn.Conv2d(FEATURES_D * 2, FEATURES_D * 4, 4, 2, 1), # 32x32x64 -> 16x16x128
            nn.LeakyReLU(0.2, inplace=True),					# Leaky ReLU
            
            nn.Conv2d(FEATURES_D * 4, FEATURES_D * 8, 4, 2, 1), # 16x16x128 -> 8x8x256
            nn.LeakyReLU(0.2, inplace=True),					# Leaky ReLU
            
            nn.Conv2d(FEATURES_D * 8, FEATURES_D * 16, 4, 2, 1), 	# 8x8x256 -> 4x4x512
            nn.LeakyReLU(0.2, inplace=True),						# Leaky ReLU
            
            nn.Conv2d(FEATURES_D * 16, 1, 4, 1, 0), 				# 4x4x512 -> 1x1x1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
if __name__ == '__main__':
    x = torch.randn(16, 3, IMG_SIZE, IMG_SIZE)

    model = Discriminator()

    out = model(x)

    print(out.shape)
