import os
import os.path as osp
import sys
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import time

import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import glob
from skimage import color
from utils import *

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Imgnet_Dataset(data.Dataset):
    def __init__(self, root,
        shuffle=False,
        mode='test',
        loader=pil_loader):

        tic = time.time()
        self.root = root
        self.loader = loader
        self.size = 224
        self.trainpath = glob.glob(root + 'train/*.JPEG')
        self.testpath = glob.glob(root + 'test/*.JPEG')

        self.path = []
        if (mode == 'train'):
            for item in self.trainpath:
                if (torch.rand(1).item()<0.002):
                    self.path.append(item)
        elif (mode == 'test'):
            for item in self.testpath:
                if (torch.rand(1).item()<0.002):
                    self.path.append(item)

        np.random.seed(0)
        if shuffle:
            perm = np.random.permutation(len(self.path))
            self.path = [self.path[i] for i in perm]

        print('Load %d images, used %fs' % (self.path.__len__(), time.time()-tic))

    def __getitem__(self, index):
        mypath = self.path[index]
        img = self.loader(mypath) # PIL Image
        img = np.array(img)
        if (img.shape[0] != self.size) or (img.shape[1] != self.size):
            img = np.array(Image.fromarray(img).resize((self.size, self.size)))

        img_lab = color.rgb2lab(np.array(img)) # np array

        img = (img - 127.5) / 127.5 # -1 to 1
        img = torch.FloatTensor(np.transpose(img, (2,0,1)))
        img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))

        img_l = torch.unsqueeze(img_lab[0],0) / 100. # L channel 0-100

        data=dict()
        
        data['A'] = img_l.view(1, 1, self.size, self.size)
        data['B'] = img.view(1, 3, self.size, self.size)
        data = add_color_patches_rand_gt(data, num_points = 25)

        returnmatrix = torch.cat((data['A'], data['hint_B'], data['mask_B']), dim=1)
        returnmatrix = returnmatrix.view(5, self.size, self.size)
        
        return returnmatrix, img


    def __len__(self):
        return len(self.path)
