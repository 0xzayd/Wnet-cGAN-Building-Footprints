import argparse

from src.models import *
from src.networks import *


import glob
import os

import numpy as np

import skimage

parser = argparse.ArgumentParser(description='arguments input')
parser.add_argument('-p','--pan', type=str, help='pan folder', required=True)
parser.add_argument('-d', '--dsm', type=str, help='dsm folder', required=True)
parser.add_argument('-l', '--label', type=str, help='label folder', required=True)

args = parser.parse_args()

# Please process all input images into small tiles of 256x256 pixels tif images
# (pan, dsm and binary label image) as shown in the example

img_height = 256
img_width = 256

pan_folder = args.pan
dsm_folder = args.dsm
label_folder = args.label

pan = []
dsm = []
label = []

pan = [skimage.io.imread(p) for p in sorted(glob.glob(pan_folder + '/*.tif'))]
dsm = [skimage.io.imread(d) for d in sorted(glob.glob(dsm_folder + '/*.tif'))]
label = [skimage.io.imread(l) for l in sorted(glob.glob(label_folder + '/*.tif'))]

pan_input = np.array(pan)
dsm_input = np.array(dsm)
label_input = np.array(label)

# training
myModel = Wnet_cgan(img_height, img_width, n_labels=1)
myModel.build_wnet_cgan([64,64,64,64],
                        (3,3), 
                        wnet_activation='selu',
                        wnet_lr=1e-4,
                        discr_inp_channels = 16,
                        discr_block_list=[32,32,32,32],
                        discr_k_size=(3,3), 
                        discr_activation='relu',
                        discr_lr=1e-4,
                        lambda_=1e-1)

myModel.fit_wnet_cgan(X=[dsm,pan,label], Y=[label,np.ones(shape=(len(label),1))])