import torch
import torch.nn as nn
import torch.nn.functional as F


class Consensus(nn.Module):
    def __init__(self, input_channels=768, num_layers=4):
        super(Consensus, self).__init__()
        self.consensus_ext = nn.ModuleList(
            [ConsensusBlock(input_channels=input_channels) for _ in range(num_layers)])

    def forward(self, x5):
        x51 = None
        for blk in self.consensus_ext:
            if x51 is None:
                x51 = blk(x5)
            else:
                x51 = x51 + blk(x51)
        consen = torch.mean(x51, [0, 2, 3], True)
        return x51 + x5 * consen


class ConsensusBlock(nn.Module):
    def __init__(self,
                 input_channels):
        super(ConsensusBlock, self).__init__()
        self.query = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Conv2d(input_channels, input_channels, 1, 1, 0)

    def correlation(self, x5, seeds):
        B, C, H5, W5 = x5.size()
        correlation_maps = torch.relu(F.conv2d(x5, weight=seeds))
        correlation_maps = correlation_maps.mean(1).view(B, -1)
        min_value = torch.min(correlation_maps, dim=1, keepdim=True)[0]
        max_value = torch.max(correlation_maps, dim=1, keepdim=True)[0]
        correlation_maps = (correlation_maps - min_value) / (max_value - min_value + 1e-12)
        correlation_maps = correlation_maps.view(B, 1, H5, W5)
        return correlation_maps

    def get_seeds(self, x_w, x_w_max, x5_norm):
        B, C, H5, W5 = x5_norm.size()
        mask = torch.zeros_like(x_w).cuda()
        mask[x_w == x_w_max] = 1
        mask = mask.view(B, 1, H5, W5)
        seeds = x5_norm * mask
        seeds = seeds.sum(3).sum(2).unsqueeze(2).unsqueeze(3)
        return seeds

    def forward(self, x5):
        B, C, H5, W5 = x5.size()
        x5 = self.conv(x5) + x5
        x_query = self.query(x5).view(B, C, -1)
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C)
        x_key = self.key(x5).view(B, C, -1)
        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1)
        x_w1 = torch.matmul(x_query, x_key) * self.scale
        x_w = x_w1.view(B * H5 * W5, B, H5 * W5)
        x_w = torch.max(x_w, -1).values
        x_w = x_w.mean(-1)
        x_w = x_w.view(B, -1)
        x_w = F.softmax(x_w, dim=-1)
        x5_norm = F.normalize(x5, dim=1)
        x_w = x_w.unsqueeze(1)
        x_w_max = torch.max(x_w, -1).values.unsqueeze(2).expand_as(x_w)
        seeds = self.get_seeds(x_w, x_w_max, x5_norm)
        cormap = self.correlation(x5_norm, seeds)
        x51 = x5 * cormap
        return x51
