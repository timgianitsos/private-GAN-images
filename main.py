import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from arg_parser import ArgParser
from layers import get_noise, Generator, Discriminator, GeneratorLoss, DiscriminatorLoss


def main():
    parser = ArgParser()
    args = parser.parse_args()

    gen = nn.DataParallel(Generator(args.latent_dim).to(args.device), args.gpu_ids)
    disc = nn.DataParallel(Discriminator().to(args.device), args.gpu_ids)

    dataset = Dataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    for epoch in range(args.num_epochs):
        for cur_step, (img, _) in enumerate(tqdm(loader, dynamic_ncols=True)):
            v = disc(img)
            breakpoint()


if __name__ == '__main__':
    main()
