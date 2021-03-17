import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset
from arg_parser import ArgParser
from logger import TrainLogger
from layers import get_noise, Generator, Discriminator, GeneratorLoss, DiscriminatorLoss


def main():
    parser = ArgParser()
    args = parser.parse_args()

    gen = nn.DataParallel(Generator(args.latent_dim).to(args.device), args.gpu_ids)
    disc = nn.DataParallel(Discriminator().to(args.device), args.gpu_ids)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=args.lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=args.lr)
    disc_loss_fn = DiscriminatorLoss()
    gen_loss_fn = GeneratorLoss()

    dataset = Dataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    logger = TrainLogger(args, len(loader), phase=None)
    logger.log_hparams(args)

    for epoch in range(args.num_epochs):
        logger.start_epoch()
        for cur_step, (img, _) in enumerate(tqdm(loader, dynamic_ncols=True)):
            logger.start_iter()

            img = img.to(args.device)

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

if __name__ == '__main__':
    main()
