import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim):
        '''
        Upscale the length, width, and depth by powers of two
        repeatedly, stopping to upscale the depth when it equals
        or exceeds `max_img_depth`, but continuing to upscale
        length and width `num_upscales` times
        '''
        super(Generator, self).__init__()
        if not ((z_dim & (z_dim - 1) == 0) and z_dim != 0):
            raise ValueError('Input `z_dim` must be a power of two')
        self.z_dim = z_dim
        interpolation = 'bilinear' #TODO parameterize this

        in_channels = z_dim
        out_channels = max(z_dim // 2, 1)
        self.blk1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=interpolation),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.blk2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=interpolation),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.blk3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=interpolation),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.blk4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3), stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.blk5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=interpolation),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.blk6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=interpolation),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.blk7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2), stride=1, padding=(1, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.blk8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=interpolation),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        in_channels = out_channels
        out_channels = max(out_channels // 2, 1)
        self.blk9 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=interpolation),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, noise):
        assert len(noise.shape) == 2
        assert noise.shape[1] == self.z_dim
        # unsqueeze to (batch_size, noise dimension, 1, 1, 1)
        x = noise.view(noise.shape[0], noise.shape[1], 1, 1)
        iterator = iter(self.modules())
        assert self is next(iterator), 'The first value from `self.modules` is self'
        for layer in iterator:
            x = layer(x)
        return x

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)
