import torch
import torch.nn as nn
from constants import *

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return nn.ReLU(inplace=True)(x + self.conv_block(x))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.project = nn.Linear(Z_DIM, IMG_SIZE * IMG_SIZE)
        self.stack = nn.Sequential(
            # Downsampling layers
            nn.Conv2d(4, FEATURES_G, 5, 1, 2),          # 128x128x4 -> 128x128x32
            nn.BatchNorm2d(FEATURES_G),                 # Batch norm
            nn.ReLU(inplace=True),                      # ReLU
            nn.MaxPool2d(2, 2),                         # 128x128x32 -> 64x64x32

            ResidualBlock(FEATURES_G),

            nn.Conv2d(FEATURES_G, FEATURES_G * 2, 5, 1, 2), # 64x64x32 -> 64x64x64
            nn.BatchNorm2d(FEATURES_G * 2),                 # Batch norm
            nn.ReLU(inplace=True),                          #ReLU
            nn.MaxPool2d(2, 2),                             # 64x64x64 -> 32x32x64
            
            ResidualBlock(FEATURES_G * 2),

            nn.Conv2d(FEATURES_G * 2, FEATURES_G * 4, 3, 1, 1), # 32x32x64 -> 32x32x128
            nn.BatchNorm2d(FEATURES_G * 4),                     # Batch norm
            nn.ReLU(inplace=True),                              # ReLU
            nn.MaxPool2d(2, 2),                                 # 32x32x128 -> 16x16x128
            
            ResidualBlock(FEATURES_G * 4),

            nn.Conv2d(FEATURES_G * 4, FEATURES_G * 8, 3, 1, 1), # 16x16x128 -> 16x16x256
            nn.BatchNorm2d(FEATURES_G * 8),                     # Batch norm
            nn.ReLU(inplace=True),                              # ReLU
            nn.MaxPool2d(2, 2),                                 # 16x16x256 -> 8x8x256
            
            ResidualBlock(FEATURES_G * 8),

            nn.Conv2d(FEATURES_G * 8, FEATURES_G * 16, 3, 1, 1),    # 8x8x256 -> 8x8x512
            nn.BatchNorm2d(FEATURES_G * 16),                        # Batch norm
            nn.ReLU(inplace=True),                                  # ReLU
            nn.MaxPool2d(2, 2),                                     # 8x8x512 -> 4x4x512
            
            ResidualBlock(FEATURES_G * 16),

            # Upsampling layers
            nn.ConvTranspose2d(FEATURES_G * 16, FEATURES_G * 8, 4, 2, 1),  # 4x4x512 -> 8x8x256
            nn.BatchNorm2d(FEATURES_G * 8),
            nn.ReLU(inplace=True),
            
            ResidualBlock(FEATURES_G * 8),

            nn.ConvTranspose2d(FEATURES_G * 8, FEATURES_G * 4, 4, 2, 1),   # 8x8x256 -> 16x16x128
            nn.BatchNorm2d(FEATURES_G * 4),
            nn.ReLU(inplace=True),
            
            ResidualBlock(FEATURES_G * 4),

            nn.ConvTranspose2d(FEATURES_G * 4, FEATURES_G * 2, 4, 2, 1),   # 16x16x128 -> 32x32x64
            nn.BatchNorm2d(FEATURES_G * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(FEATURES_G * 2, 3, 4, 2, 1),                # 32x32x64 -> # 64x64x3 (as the patch size)
            nn.Sigmoid()                                                   # Final activation
        )

    def forward(self, x, z):
        b_size = x.shape[0]
        cond = self.project(z) # Linear projection (z -> cond)
        cond = torch.reshape(cond, (b_size, 1, IMG_SIZE, IMG_SIZE)) # cond -> 128x128x1
        x = torch.concat((cond, x), dim=1)  # 128x128x1 + 128x128x3 -> 128x128x4
        return self.stack(x)

if __name__ == '__main__':
    model = Generator()

    x = torch.randn(16, 3, IMG_SIZE, IMG_SIZE)
    z = torch.randn(16, Z_DIM)

    images = model(x, z)

    print(images.shape) 
