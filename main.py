import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from arg_parser import ArgParser
from layers import Generator, get_noise
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = ArgParser()
    args = parser.parse_args()

    gen = nn.DataParallel(Generator(args.latent_dim).to(args.device), args.gpu_ids)

    dataset = Dataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    for epoch in range(args.num_epochs):
        for cur_step, (img, _) in enumerate(tqdm(loader, dynamic_ncols=True)):
            fake_noise = get_noise(args.batch_size, args.latent_dim, device=args.device)
            v = gen(fake_noise)
            plt.imshow(img[0,0], cmap='Greys')
            plt.show()


if __name__ == '__main__':
    main()
