import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.model import BaseModule

class up_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding='same')
        self.us = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, input):
        return self.us(self.conv(input))
@MODELS.register_module()
class TransmitterDecoder(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.up1 = up_block(dim, dim//2)
        self.up2 = up_block(dim//2, dim//4)
        self.up3 = up_block(dim//4, dim//8)
        self.up4 = up_block(dim//8, dim//16)
        self.cls_head = nn.Conv2d(in_channels=dim//16, out_channels=num_classes, kernel_size=1, stride=1)

    def forward(self, input):
        x1 = self.up1(input)
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        x4 = self.up4(x3)
        cls = self.cls_head(x4)
        return cls