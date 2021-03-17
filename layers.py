import torch
import torch.nn as nn

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

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
        interpolation = 'bilinear'  # TODO parameterize this

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
        x = noise.view(*noise.shape, 1, 1)
        x1 = self.blk1(x)
        x2 = self.blk2(x1)
        x3 = self.blk3(x2)
        x4 = self.blk4(x3)
        x5 = self.blk5(x4)
        x6 = self.blk6(x5)
        x7 = self.blk7(x6)
        x8 = self.blk8(x7)
        x9 = self.blk9(x8)

        return x9

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.rl = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.mp1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.mp2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.mp3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.mp4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv5 = nn.Conv2d(32, 64, kernel_size=(2, 4), stride=(1, 1), padding=(1, 3))
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.mp6 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.mp7 = nn.MaxPool2d(kernel_size=(1,2), stride=(2,2))

        self.conv8 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(1)
        self.mp8 = nn.MaxPool2d(kernel_size=(1,2), stride=(2,2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.rl(x)
        x = self.mp3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.rl(x)
        x = self.mp4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.rl(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.rl(x)
        x = self.mp6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.rl(x)
        x = self.mp7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.rl(x)
        x = self.mp8(x)

        return x

class GeneratorLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, real, fake, disc):
        disc_fake_pred = disc(fake)
        gen_adv_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_adv_loss

class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, real, fake, disc):
        disc_fake_pred = disc(fake.detach())
        disc_fake_adv_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_adv_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_adv_loss = (disc_fake_adv_loss + disc_real_adv_loss) / 2
        return disc_adv_loss
