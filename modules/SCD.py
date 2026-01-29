import torch
import torch.nn as nn
import torch.nn.functional as F


class SCD(nn.Module):

    def __init__(self, in_channels, encoder_kernel=5, kernel_size=3, stride=1, use_consensus=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.encoder_kernel = encoder_kernel
        self.pad = (kernel_size - 1) // 2
        self.stride = stride
        self.in_channels = in_channels
        self.use_consensus = use_consensus
        self.gen_kernel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=self.kernel_size ** 2, stride=1,
                      kernel_size=self.encoder_kernel, bias=False, padding=int((self.encoder_kernel - 1) / 2))
        )

        if use_consensus:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.t_c = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels // 4, kernel_size=1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(in_channels=self.in_channels // 4, out_channels=self.in_channels, kernel_size=1, bias=False),
                nn.ReLU(True),
            )

    def forward(self, x):
        b, c, h, w = x.shape
        oh, ow = (h - 1) // self.stride + 1, (w - 1) // self.stride + 1
        weight = self.gen_kernel(x)
        weight = weight.reshape(b, 1, self.kernel_size ** 2, oh, ow).repeat(1, c, 1, 1, 1)

        if self.use_consensus:
            x_con = torch.mean(self.avg(x), dim=0, keepdim=True)
            tmp = self.t_c(x_con).reshape(1, c, 1, 1, 1)
            weight = weight * tmp
        weight = weight.permute(0, 1, 3, 4, 2).softmax(dim=-1)
        weight = weight.reshape(b, c, oh, ow, self.kernel_size, self.kernel_size)

        pad_x = F.pad(x, pad=[self.pad] * 4, mode='reflect')
        pad_x = pad_x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        pad_x = pad_x.reshape(b, c, oh, ow, self.kernel_size, self.kernel_size)
        out = weight * pad_x
        out = out.sum(dim=(-1, -2)).reshape(b, c, oh, ow)
        return out
