import argparse

import rasterio

import osr
import ogr
import gdal
from osgeo import gdal_array

from src.models import *
from src.networks import *
from src.data import Data
from src.utils import *

parser = argparse.ArgumentParser(description='arguments input')
parser.add_argument('-d','--dsm', type=str, help='Path to DSM tif file', required=True)
parser.add_argument('-p','--pan', type=str, help='Path to PAN tif file', required=True)
parser.add_argument('-w','--weights', type=str, help='Path to HDF5 weights file', required=True)
parser.add_argument('-o','--output', type=str, help='Path to tif Output file', required=True)


args = parser.parse_args()


dsm_path = args.dsm
pan_path = args.pan
weights = args.weights
output = args.output

# padding of 16px here to avoid artefacts
pad = (16,16)
shape = (224,224) # 256 - 2 * 16

# Processing input
ary_dsm = np.moveaxis(rasterio.open(dsm_path).read(),0,-1)
ary_pan = np.moveaxis(rasterio.open(pan_path).read(),0,-1)
print('Input dsm has shape:', ary_dsm.shape)
print('Input pan has shape:', ary_pan.shape)

# Padding via mirroring to avoid artefacts
padH = shape[0]*((ary_dsm.shape[0]//shape[0])+min(1,(ary_dsm.shape[0]%shape[0])))
padW = shape[1]*((ary_dsm.shape[1]//shape[1])+min(1,(ary_dsm.shape[1]%shape[1])))

h_pad = (pad[0],padH - ary_dsm.shape[0] + pad[0])
w_pad = (pad[1],padW - ary_dsm.shape[1] + pad[1])

padded_ary_dsm = np.pad(ary_dsm, pad_width=(h_pad, w_pad, (0, 0)), mode='symmetric')
padded_ary_pan = np.pad(ary_pan, pad_width=(h_pad, w_pad, (0, 0)), mode='symmetric')

W_dsm, _ = ary_to_tiles_forward(padded_ary_dsm, ary_dsm, pad=pad, shape=shape)
W_pan, _ = ary_to_tiles_forward(padded_ary_pan, ary_pan, pad=pad, shape=shape)



# Inference in NN
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

myModel.wnet_cgan.load_weights(weights)

W_hat_test = myModel.wnet.predict([W_dsm, W_pan], verbose=1)
print('Finished predict on {0} tiles of shape ({1},{2}) for: {4}'.format(*W_hat_test.shape + (dsm_path,)))


# processing into tif file
W_hat_ary = tiles_to_ary_forward(stacked_ary=W_hat_test, pad=pad, padded_ary_shape_padded=padded_ary_dsm.shape[:2])

if output != None:
    meta = rasterio.open(pan_path).meta.copy()
    meta.update({'count': 1,
                'dtype':'float32'})
    
    sc = MinMaxScaler(copy=True, feature_range=(0, 1)).fit(W_hat_ary[:,:,0])
    pp_ = sc.transform(W_hat_ary[:,:,0])
 
    with rasterio.open(output, 'w', **meta) as rr:
        rr.write(pp_[pad[0]:meta['height']+pad[0], pad[1]:meta['width']+pad[1]].astype(np.float32),1)