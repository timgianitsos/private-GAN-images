import os
from os.path import dirname, join

import torch
import torch.utils.data

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
	return torch.tensor(raster, dtype=torch.uint8)

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self):
        dataset_path = join(os.curdir, dirname(__file__), 'orl-database-of-faces')
        # Sort by length first, then lexicographically
        dirnames = sorted([e for e in os.scandir(dataset_path) if e.is_dir()], key=lambda e: (len(e.name), e.name))
        self.filenames = [(filename.path, label) for label, e in enumerate(dirnames) for filename in os.scandir(e) if filename.name.endswith(f'{os.extsep}pgm')]

    def __getitem__(self, index):
        # specify path relative to this script
        label, entry = self.filenames[index]
        return label, read_pgm(open(entry, 'rb'))
