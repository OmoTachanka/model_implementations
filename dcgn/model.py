import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, c, feats_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(c, feats_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(feats_d, feats_d*2, 4, 2, 1),
            self._block(feats_d*2, feats_d*4, 4, 2, 1),
            self._block(feats_d*4, feats_d*8, 4, 2, 1),
            nn.Conv2d(feats_d*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, feats_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, feats_g*16, 4, 1, 0),
            self._block(feats_g*16, feats_g*8, 4, 2, 1),
            self._block(feats_g*8, feats_g*4, 4, 2, 1),
            self._block(feats_g*4, feats_g*2, 4, 2, 1),
            nn.ConvTranspose2d(feats_g* 2, channels_img, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    
    def _block(self, in_channels, out_channels, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)



def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn(N, in_channels, H, W)
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn(N, z_dim, 1, 1)
    assert gen(z).shape == (N, in_channels, H, W)
    print("SUS")

# test()