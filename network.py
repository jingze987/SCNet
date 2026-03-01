import torch
import torch.nn as nn
from modules.decoder import FPNDecoder
from modules.consensus import Consensus
from modules.encoder import SwinTransformer
from modules.SCL import SCLBlock


class SODNet(nn.Module):
    def __init__(self):
        super(SODNet, self).__init__()
        self.SCL = SCLBlock(kernel_nums=8, kernel_size=3)
        self.backbone = SwinTransformer(img_size=256, drop_path_rate=0.2, embed_dim=96, \
                                        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=16)
        self.consensus = Consensus(input_channels=768, num_layers=4)
        self.decoder = FPNDecoder()

    def forward(self, x):
        x_scl, x_edge = self.SCL(x)
        x_fpn = self.backbone(x_scl)
        x1, x2, x3, x4 = x_fpn[0], x_fpn[1], x_fpn[2], x_fpn[3]
        x5 = self.consensus(x4)
        preds = self.decoder(x5, x3, x2, x1)
        return preds