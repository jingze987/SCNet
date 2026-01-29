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

if __name__ == "__main__":
    import torch
    from thop import profile
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    net = SODNet().cuda()
    x = (torch.randn(1, 3, 256, 256).cuda(), )
    flops = FlopCountAnalysis(net, x)
    print("FLOPs: {}".format(flops.total()))
    print(parameter_count_table(net))
    # SCL = SCLBlock(kernel_nums=8, kernel_size=3).cuda()
    # backbone = SwinTransformer(img_size=256, drop_path_rate=0.2, embed_dim=96, \
    #                            depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=16).cuda()
    # consensus = Consensus(input_channels=768, num_layers=4).cuda()
    # decoder = FPNDecoder().cuda()
    # x1 = torch.randn(1, 3, 256, 256).cuda()
    # macs, params = profile(SCL, inputs=(x1, ), verbose=False)
    # params_M = params / 1e6
    # flops = macs * 2  # 1 MAC ≈ 2 FLOPs
    # gflops = flops / 1e9  # 常见论文口径：GFLOPs
    # print(f"SCL Params: {params_M:.3f} M")
    # print(f"SCL FLOPs : {gflops:.3f} GFLOPs")  # 你也可以改成 MFLOPs / FLOPs
    # x2 = torch.randn(1, 3, 256, 256).cuda()
    # macs, params = profile(backbone, inputs=(x2, ), verbose=False)
    # params_M = params / 1e6
    # flops = macs * 2  # 1 MAC ≈ 2 FLOPs
    # gflops = flops / 1e9  # 常见论文口径：GFLOPs
    # print(f"BB Params: {params_M:.3f} M")
    # print(f"BB FLOPs : {gflops:.3f} GFLOPs")  # 你也可以改成 MFLOPs / FLOPs
    # x3 = torch.randn(1, 768, 8, 8).cuda()
    # macs, params = profile(consensus, inputs=(x3, ), verbose=False)
    # params_M = params / 1e6
    # flops = macs * 2  # 1 MAC ≈ 2 FLOPs
    # gflops = flops / 1e9  # 常见论文口径：GFLOPs
    # print(f"Consensus Params: {params_M:.3f} M")
    # print(f"Consensus FLOPs : {gflops:.3f} GFLOPs")  # 你也可以改成 MFLOPs / FLOPs
    # x4 = torch.randn(1, 768, 8, 8).cuda()
    # x3 = torch.randn(1, 768, 8, 8).cuda()
    # x2 = torch.randn(1, 384, 16, 16).cuda()
    # x1 = torch.randn(1, 192, 32, 32).cuda()
    # macs, params = profile(decoder, inputs=(x4, x3, x2, x1), verbose=False)
    # params_M = params / 1e6
    # flops = macs * 2  # 1 MAC ≈ 2 FLOPs
    # gflops = flops / 1e9  # 常见论文口径：GFLOPs
    # print(f"decoder Params: {params_M:.3f} M")
    # print(f"decoder FLOPs : {gflops:.3f} GFLOPs")  # 你也可以改成 MFLOPs / FLOPs