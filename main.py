import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dataset
from arg_parser import ArgParser

def main():
    parser = ArgParser()
    args = parser.parse_args()
    dataset = Dataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

if __name__ == '__main__':
    main()
