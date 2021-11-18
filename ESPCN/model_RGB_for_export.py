import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()
        r = upscale_factor
        # feature map layer
        self.feature_map = nn.Sequential(nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2)),
                                         nn.Tanh(),
                                         nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
                                         nn.Tanh())
        # sub-pixel convolution layer
        self.sub_pixel_layer = nn.Sequential(nn.Conv2d(32, 3 * upscale_factor ** 2, (3, 3), (1, 1), (1, 1)),
                                             nn.PixelShuffle(upscale_factor))

    def forward(self, x):
        x = self.feature_map(x)
        x = self.sub_pixel_layer(x)
        x = x.permute(0, 2, 3, 1) # this line only for export
        return x


if __name__ == '__main__':
    import torch

    model = ESPCN(3)
    print(model)

    sample = torch.rand((1, 3, 333, 333))
    output = model(sample)
    print(output)