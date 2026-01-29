import torch.nn.functional as F
import torch.nn as nn
import torch


class CCRConvolution(nn.Module):

    def __init__(self, kernel_nums=8, kernel_size=3):
        super(CCRConvolution, self).__init__()

        self.kernel_nums = kernel_nums
        self.kernel_size = kernel_size
        self.rg_bn = nn.BatchNorm2d(kernel_nums)
        self.gb_bn = nn.BatchNorm2d(kernel_nums)
        self.rb_bn = nn.BatchNorm2d(kernel_nums)
        self.filter = torch.nn.Parameter(torch.randn(self.kernel_nums, 1, self.kernel_size, self.kernel_size))

    def mean_constraint(self, kernel):
        bs, cin, kw, kh = kernel.shape
        kernel_mean = torch.mean(kernel.view(bs, -1), dim=1, keepdim=True)
        kernel = (kernel.view(bs, -1) - kernel_mean)
        return kernel.view(bs, cin, kw, kh)

    def forward(self, img):
        zeroMasks = torch.zeros_like(img)
        zeroMasks[img == 0] = 1
        log_img = torch.log(img + 1e-7)

        red_chan = log_img[:, 0, :, :].unsqueeze(1)
        green_chan = log_img[:, 1, :, :].unsqueeze(1)
        blue_chan = log_img[:, 2, :, :].unsqueeze(1)
        normalized_filter = self.mean_constraint(self.filter)

        filt_r1 = F.conv2d(red_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_g1 = F.conv2d(green_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_rg = filt_r1 + filt_g1
        filt_rg = self.rg_bn(filt_rg)

        filt_g2 = F.conv2d(green_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_b1 = F.conv2d(blue_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_gb = filt_g2 + filt_b1
        filt_gb = self.gb_bn(filt_gb)

        filt_r2 = F.conv2d(red_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_b2 = F.conv2d(blue_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_rb = filt_r2 + filt_b2
        filt_rb = self.rb_bn(filt_rb)

        rg = filt_rg
        rg = torch.where(zeroMasks[:, 0:1, ...].expand(-1, self.kernel_nums, -1, -1) == 1, 0, rg)
        gb = filt_gb
        gb = torch.where(zeroMasks[:, 1:2, ...].expand(-1, self.kernel_nums, -1, -1) == 1, 0, gb)
        rb = filt_rb
        rb = torch.where(zeroMasks[:, 2:3, ...].expand(-1, self.kernel_nums, -1, -1) == 1, 0, rb)

        out = torch.cat([rg, gb, rb], dim=1)

        return out


class SCLBlock(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3):
        super(SCLBlock, self).__init__()

        self.proj = nn.Sequential(*[nn.Conv2d(3, 24, 3, 1, 1, groups=1),
                                    nn.BatchNorm2d(24),
                                    nn.LeakyReLU(),
                                    ])
        self.fuse_blk = nn.Sequential(*[nn.Conv2d(48, 32, 3, 1, 1, groups=2),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(32, 3, 3, 1, 1, groups=1)])
        self.CCR = CCRConvolution(kernel_nums=kernel_nums, kernel_size=kernel_size)
        self.SPH = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(6, 24, 3, 1, 1),
            nn.Conv2d(24, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        self.pred_head = nn.Sequential(
            nn.Conv2d(3, 1, 3, 1, 1),
        )

    def forward(self, x):
        feat_reflect = self.CCR(x)
        feat_proj = self.proj(x)

        feats_ = torch.concat((feat_proj, feat_reflect), dim=1)
        x_fuse = self.fuse_blk(feats_)
        x_struct = self.SPH(x_fuse)
        x_con = torch.mean(x_struct, dim=0, keepdim=True).expand(x_struct.size(0), -1, -1, -1)
        x_gate = self.gate(torch.cat([x_struct, x_con], dim=1)) * x_struct
        x_out = x_gate + x_fuse
        x_pred = self.pred_head(x_gate)

        return x_out, x_pred
