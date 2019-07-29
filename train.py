import argparse

from src.models import *
from src.networks import *
from src.Data import Data
from src.DataGeneration import *

parser = argparse.ArgumentParser(description='arguments input')
parser.add_argument('-d','--data', type=str, help='Training data folder with subfolders: DSM, PAN, LABEL', required=True)


args = parser.parse_args()

img_height = 256
img_width = 256

data_folder = args.data


data = Data(data_folder)

# split to train, validation, test
dsm_train, pan_train, label_train, dsm_vld, pan_vld, label_vld, dsm_tst, pan_tst, label_tst = data.split_trn_vld_tst()

# create two generators: for training and for validation
train_gen = DataGenerator(dsm_train, pan_train, label_train, pred_fn=None, batch_size=8, 
                          shuffle=True)
valid_gen = DataGenerator(dsm_vld, pan_vld, label_vld, pred_fn=None, batch_size=8, 
                          shuffle=True)

# training
myModel = Wnet_cgan(img_height, img_width, n_labels=1)
myModel.build_wnet_cgan([64,64,128,128],
                        (3,3), 
                        wnet_activation='selu',
                        wnet_lr=1e-4,
                        discr_inp_channels = 16,
                        discr_block_list=[32,32,64,64],
                        discr_k_size=(3,3), 
                        discr_activation='relu',
                        discr_lr=1e-4,
                        lambda_=1e-1)

myModel.fit_wnet_cgan(train_gen, valid_gen, adv_epochs=50, gen_epochs=100,
                     adv_steps_epoch=50, gen_steps_epoch=100, n_rounds=80)
