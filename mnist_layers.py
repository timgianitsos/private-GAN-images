import torch
import torch.nn as nn

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating a noise vector: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
    n_samples: the number of samples in the batch, a scalar
    z_dim: the dimension of the noise vector, a scalar
    device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)

class Generator(nn.Module):
    '''
    Generator Class
    Values:
    z_dim: the dimension of the noise vector, a scalar
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white, so that's our default
    hidden_dim: the inner dimension, a scalar
    '''
    
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of the DCGAN, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation
        Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        '''
        # Hints: You'll find nn.ConvTranspose2d (https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose2d.html) 
        #    and nn.BatchNorm2d (https://pytorch.org/docs/master/generated/torch.nn.BatchNorm2d.html) useful!
        #    Steps:
        #    1) Do a transposed convolution using the given parameters
        #    2) Do a batchnorm, except for the last layer.
        #    3) Follow each batchnorm with a ReLU activation.
        #    4) If its the final layer, use a Tanh activation after the deconvolution instead of a ReLU.

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the Generator: Given a noise vector, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
        noise: a noise tensor with dimensions (batch_size, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the Generator: Given a noise vector, 
        returns a generated image.
        Parameters:
        noise: a noise tensor with dimensions (batch_size, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white (1 channel), so that's our default.
    hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation
        Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        '''

        # Hints: You'll find nn.Conv2d (https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html),
        #    nn.BatchNorm2d (https://pytorch.org/docs/master/generated/torch.nn.BatchNorm2d.html), and
        #    nn.LeakyReLU (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html) useful!
        #    Steps:
        #    1) Add a convolutional layer using the given parameters
        #    2) Do a batchnorm, except for the last layer.
        #    3) Follow each batchnorm with a LeakyReLU activation with slope 0.2.
        
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the Discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
        image: a flattened image tensor with dimension (im_dim)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

class GeneratorLoss(nn.Module):
    """Implementations of various losses for the generator"""

    def __init__(self, lambda_adversarial=1.):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.lambda_adversarial = lambda_adversarial

    def bce_adversarial_loss(self, real, fake, disc):
        disc_fake_pred = disc(fake)
        gen_adv_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_adv_loss

    def lsgan_adversarial_loss(self, real, fake, disc):
        """Least-squares loss, as defined in https://arxiv.org/abs/1611.04076 (Mao et al. 2016)"""
        disc_fake_pred = disc(fake)
        gen_adv_loss = torch.mean((disc_fake_pred - 1.) ** 2)
        return gen_adv_loss

    def hinge_adversarial_loss(self, real, fake, disc):
        """Hinge loss, as defined in https://arxiv.org/abs/1705.02894v2 (Lim and Ye 2017)"""
        disc_fake_pred = disc(fake)
        gen_adv_loss = torch.mean(-disc_fake_pred)

    #### IMPLEMENT YOUR LOSS FUNCTIONS HERE ####
    #### IMPLEMENT YOUR LOSS FUNCTIONS HERE ####

    def forward(self, real, fake, disc):
        gen_loss = 0
        gen_loss += self.lambda_adversarial * self.bce_adversarial_loss(real, fake, disc)
        #### CALL YOUR LOSS FUNCTIONS HERE ####
        #### CALL YOUR LOSS FUNCTIONS HERE ####
        return gen_loss

class DiscriminatorLoss(nn.Module):
    """Implementations of various losses for the discriminator"""

    def __init__(self, lambda_adversarial=1):
        """
        """
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.lambda_adversarial = lambda_adversarial

    def bce_adversarial_loss(self, real, fake, disc):
        disc_fake_pred = disc(fake.detach())
        disc_fake_adv_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_adv_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_adv_loss = (disc_fake_adv_loss + disc_real_adv_loss) / 2
        return disc_adv_loss

    def lsgan_adversarial_loss(self, real, fake, disc):
        """Least-squares loss, as defined in https://arxiv.org/abs/1611.04076 (Mao et al. 2016)"""
        disc_fake_pred = disc(fake.detach())
        disc_fake_adv_loss = torch.mean(disc_fake_pred ** 2)
        disc_real_pred = disc(real)
        disc_real_adv_loss = torch.mean((disc_real_pred - 1.) ** 2)
        disc_adv_loss = (disc_fake_adv_loss + disc_real_adv_loss) / 2
        return disc_adv_loss

    def hinge_adversarial_loss(self, real, fake, disc):
        """Hinge loss, as defined in https://arxiv.org/abs/1705.02894v2 (Lim and Ye 2017)"""
        disc_fake_pred = disc(fake.detach())
        disc_fake_adv_loss = torch.mean(F.relu(1 - disc_fake_pred))
        disc_real_pred = disc(real)
        disc_fake_adv_loss = torch.mean(F.relu(1 + disc_fake_pred))
        disc_adv_loss = (disc_fake_adv_loss + disc_real_adv_loss) / 2
        return disc_adv_loss

    #### IMPLEMENT YOUR LOSS FUNCTIONS HERE ####
    #### IMPLEMENT YOUR LOSS FUNCTIONS HERE ####

    def forward(self, real, fake, disc):
        disc_loss = 0
        disc_loss += self.lambda_adversarial * self.bce_adversarial_loss(real, fake, disc)
        #### CALL YOUR LOSS FUNCTIONS HERE ####
        #### CALL YOUR LOSS FUNCTIONS HERE ####
        return disc_loss
