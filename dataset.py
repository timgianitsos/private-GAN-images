import os
from os.path import dirname, join

import torch
import torch.utils.data
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, Compose
from torchvision import transforms
from PIL import Image


def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    # https://stackoverflow.com/a/35726744/7102572
    assert pgmf.readline().decode('utf-8') == 'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return torch.tensor(raster, dtype=torch.uint8).unsqueeze(0).float()


class Dataset(torch.utils.data.Dataset):
    transforms = Compose([
        lambda img_tensor: img_tensor / 255,
        RandomHorizontalFlip(),
        RandomRotation(10, resample=2)
    ])

    def __init__(self):
        dataset_path = join(os.curdir, dirname(__file__), 'orl-database-of-faces')

        # Sort by length first, then lexicographically
        dirnames = sorted([e for e in os.scandir(dataset_path) if e.is_dir()], key=lambda e: (len(e.name), e.name))
        self.filenames = [
            (filename.path, label)
            for label, e in enumerate(dirnames)
            for filename in sorted(os.scandir(e), key=lambda e: (len(e.name), e.name))
            if filename.name.endswith(f'{os.extsep}pgm')
        ]

    def __getitem__(self, index):
        # Specify path relative to this script
        entry, label = self.filenames[index]
        img: torch.Tensor = read_pgm(open(entry, 'rb'))
        return self.transforms(img), label

    def __len__(self):
        return len(self.filenames)


class MNISTDataset(torch.utils.data.Dataset):
    """Dataset class for MNIST"""

    def __init__(self, data_root='./mnist_png/training', num=None, normalize=True, rotate=False):

        # Recursively exract paths to all .png files in subdirectories
        assert num is None or num in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.file_paths = []
        for path, subdirs, files in os.walk(data_root):
            for name in files:
                if name.endswith(".png") and (num is None or str(num) in path):
                    self.file_paths.append(os.path.join(path, name))

        self.transform = self._set_transforms(normalize, rotate)

    def _set_transforms(self, normalize, rotate):
        """Decide transformations to data to be applied"""
        transforms_list = []

        # Normalize to the mean and standard deviation all pretrained
        # torchvision models expect
        normalize = transforms.Normalize(mean=(0.5,),
                                         std=(0.5,))

        # 1) transforms PIL image in range [0,255] to [0,1],
        # 2) transposes [H, W, C] to [C, H, W]
        if normalize:
            transforms_list += [transforms.ToTensor(), normalize]
        else:
            transforms_list += [transforms.ToTensor()]

        # Applies a random rotation augmentation
        if rotate:
            transforms_list += [transforms.RandomRotation(30)]

        transform = transforms.Compose([t for t in transforms_list if t])
        return transform

    def __len__(self):
        """Required: specify dataset length for dataloader"""
        return len(self.file_paths)

    def __getitem__(self, index):
        """Required: specify what each iteration in dataloader yields"""
        img = Image.open(self.file_paths[index])
        img = self.transform(img)
        return img

def main():
    from matplotlib import pyplot as plt
    d = Dataset()
    while True:
        i = int(input('Enter an index: '))
        f, label = d[i]
        print(f'Displaying: {d.filenames[i][0]} with label {label}')
        plt.imshow(f, plt.cm.gray)
        plt.show()


if __name__ == '__main__':
    main()
