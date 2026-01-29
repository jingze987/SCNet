import torch.nn as nn


class EnLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x


class LatLayer(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class FPNDecoder(nn.Module):
    def __init__(self):
        super(FPNDecoder, self).__init__()
        self.toplayer = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0))
        self.latlayer3 = LatLayer(in_channel=768, mid_channel=256)
        self.latlayer2 = LatLayer(in_channel=384, mid_channel=256)
        self.latlayer1 = LatLayer(in_channel=192, mid_channel=128)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                                        kernel_size=2, stride=2, padding=0),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                                        kernel_size=2, stride=2, padding=0),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                                        kernel_size=2, stride=2, padding=0),
                                     )
        self.enlayer3 = EnLayer()
        self.enlayer2 = EnLayer()
        self.enlayer1 = EnLayer()
        self.dslayer = DSLayer()

    def forward(self, weighted_x5, x3, x2, x1):
        preds = []
        p4 = self.toplayer(weighted_x5) + self.latlayer3(x3)
        p3 = self.enlayer3(p4)
        p2 = self.deconv3(p3) + self.latlayer2(x2)
        p2 = self.enlayer2(p2)
        p1 = self.deconv2(p2) + self.latlayer1(x1)
        p1 = self.enlayer1(p1)
        preds.append(self.dslayer(self.deconv1(p1)))
        return preds
