import torch
import torch.nn as nn

class CNNB(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2):
        super(CNNB, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self._block(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, feats = [64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, feats[0], 4, stride = 2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = feats[0]
        for feat in feats[1:]:
            layers.append(
                CNNB(in_channels, feat, stride = 1 if feat == feats[-1] else 2)
            )
            in_channels = feat

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim = 1)
        return self.model(self.initial(x))


def test():
    x = torch.rand((32, 3, 256, 256))
    y = torch.rand((32, 3, 256, 256))
    model = Discriminator()
    op = model(x, y)
    print(op.shape)
    print(model)

# if __name__ == "__main__":
#     test()

