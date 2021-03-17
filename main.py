import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset, MNISTDataset
from arg_parser import ArgParser
from logger import TrainLogger
# from layers import get_noise, Generator, Discriminator, GeneratorLoss, DiscriminatorLoss
from mnist_layers import get_noise, Generator, Discriminator, GeneratorLoss, DiscriminatorLoss, weights_init


def lr_lambda(num_epochs):
    def lr_lambda_inner(epoch, lr_decay_after=10):
        """ Function for scheduling learning """
        if epoch < lr_decay_after:
            return 1.
        else:
            return 1 - float(epoch - lr_decay_after) / (
                    num_epochs - lr_decay_after + 1e-8)

    return lr_lambda_inner


def main():
    parser = ArgParser()
    args = parser.parse_args()
    beta_1 = 0.5
    beta_2 = 0.999

    gen = Generator(args.latent_dim).to(args.device)
    gen = gen.apply(weights_init)
    gen = nn.DataParallel(gen, args.gpu_ids)
    disc = Discriminator().to(args.device)
    disc = disc.apply(weights_init)
    disc = nn.DataParallel(disc, args.gpu_ids)

    gen_opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(beta_1, beta_2))
    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_opt, lr_lambda=lr_lambda(args.num_epochs))
    disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_opt, lr_lambda=lr_lambda(args.num_epochs))
    disc_loss_fn = DiscriminatorLoss()
    gen_loss_fn = GeneratorLoss()

    # dataset = Dataset()
    dataset = MNISTDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    loaded_dataset = [t.to(args.device) for t in loader]
    logger = TrainLogger(args, len(loader), phase=None)
    logger.log_hparams(args)

    for epoch in range(args.num_epochs):
        logger.start_epoch()
        # for cur_step, (img, _) in enumerate(tqdm(loader, dynamic_ncols=True)):
        for cur_step, img in enumerate(tqdm(loaded_dataset, dynamic_ncols=True)):
            logger.start_iter()

            disc_opt.zero_grad()
            fake_noise = get_noise(args.batch_size, args.latent_dim, device=args.device)
            fake = gen(fake_noise)
            disc_loss = disc_loss_fn(img, fake, disc)
            disc_loss.backward()
            disc_opt.step()

            gen_opt.zero_grad()
            fake_noise_2 = get_noise(args.batch_size, args.latent_dim, device=args.device)
            fake_2 = gen(fake_noise_2)
            gen_loss = gen_loss_fn(img, fake_2, disc)
            gen_loss.backward()
            gen_opt.step()

            logger.log_iter_gan_from_latent_vector(img, fake, gen_loss, disc_loss)
            logger.end_iter()

        logger.end_epoch()
        gen_scheduler.step()
        disc_scheduler.step()


if __name__ == '__main__':
    main()
