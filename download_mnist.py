import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid
import torch.utils.data as data

import os
import tarfile
import urllib.request
import random
from PIL import Image


thetarfile = 'https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz'
ftpstream = urllib.request.urlopen(thetarfile)
thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
thetarfile.extractall()

