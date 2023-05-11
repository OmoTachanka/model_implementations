import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down = True, act ="relu", use_dropout = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x



class Generator(nn.Module):
    def __init__(self, in_channels = 3, feats = 64):
        super().__init__()
        self.init_gen = nn.Sequential(
            nn.Conv2d(in_channels, feats, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        ) #128

        self.ds1 = Block(feats, feats*2, down = True, act="leaky", use_dropout=False)#64
        self.ds2 = Block(feats*2, feats*4, down = True, act="leaky", use_dropout=False)#32
        self.ds3 = Block(feats*4, feats*8, down = True, act="leaky", use_dropout=False)#16
        self.ds4 = Block(feats*8, feats*8, down = True, act="leaky", use_dropout=False)#8
        self.ds5 = Block(feats*8, feats*8, down = True, act="leaky", use_dropout=False)#4
        self.ds6 = Block(feats*8, feats*8, down = True, act="leaky", use_dropout=False)#2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats*8, feats*8, 4, 2, 1),#1
            nn.ReLU()
        )

        self.us1 = Block(feats*8, feats*8, down = False, act="relu", use_dropout=True)#2
        self.us2 = Block(feats*8*2, feats*8, down = False, act="relu", use_dropout=True)#4
        self.us3 = Block(feats*8*2, feats*8, down = False, act="relu", use_dropout=True)#8
        self.us4 = Block(feats*8*2, feats*8, down = False, act="relu", use_dropout=True)#16
        self.us5 = Block(feats*8*2, feats*4, down = False, act="relu", use_dropout=True)#32
        self.us6 = Block(feats*4*2, feats*2, down = False, act="relu", use_dropout=True)#64
        self.us7 = Block(feats*2*2, feats, down = False, act="relu", use_dropout=True)#128

        self.fin_us = nn.Sequential(
            nn.ConvTranspose2d(feats*2, in_channels, 4, 2, 1),#256
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.init_gen(x)
        d2 = self.ds1(d1)
        d3 = self.ds2(d2)
        d4 = self.ds3(d3)
        d5 = self.ds4(d4)
        d6 = self.ds5(d5)
        d7 = self.ds6(d6)
        bottleneck = self.bottleneck(d7)
        u1 = self.us1(bottleneck)
        u2 = self.us2(torch.cat([u1, d7], dim = 1))
        u3 = self.us3(torch.cat([u2, d6], dim = 1))
        u4 = self.us4(torch.cat([u3, d5], dim = 1))
        u5 = self.us5(torch.cat([u4, d4], dim = 1))
        u6 = self.us6(torch.cat([u5, d3], dim = 1))
        u7 = self.us7(torch.cat([u6, d2], dim = 1))
        return self.fin_us(torch.cat([u7, d1], dim = 1))


def test():
    x = torch.rand((32, 3, 256, 256))
    model = Generator()
    op = model(x)
    print(op.shape)

# if __name__ == "__main__":
#     test()
