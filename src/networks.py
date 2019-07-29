from src.utils import make_trainable

import keras

from keras.optimizers import adam
from keras.layers import Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import Input, AveragePooling2D, MaxPooling2D, Dropout, Lambda, AlphaDropout
from keras.layers.merge import concatenate
from keras import backend as K
from keras.models import Model


def Wnet(inp_dsm, inp_pan, blocks_list, k_size, activation, n_labels=1, name=None):
    """
    input:
        n_labels, int, number of labels = 1
        blocks list, list, number of filters in each block
        k_size, tuple, filter size
        activation, string, activation function
        
    output:
        keras model
    """
    
    
    # PAN
    
    k_init = 'lecun_normal'

    if K.image_data_format() == 'channels_first':
        concat_axis = 1
    else:
        concat_axis = 3
        
    encoder_pan = inp_pan
    
    list_encoders = []
    
    print('Building Unet for PAN Image')
    print(blocks_list)   
    
    with K.name_scope('PAN_UNet'):
        for l_idx, n_ch in enumerate(blocks_list):
            with K.name_scope('Encoder_block_{0}'.format(l_idx)):
                encoder_pan = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(encoder_pan)
                encoder_pan = AlphaDropout(0.1*l_idx, )(encoder_pan)
                encoder_pan = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 dilation_rate=(2, 2),
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(encoder_pan)
                list_encoders.append(encoder_pan)
                # add maxpooling layer except the last layer
                if l_idx < len(blocks_list) - 1:
                    encoder_pan = MaxPooling2D(pool_size=(2,2))(encoder_pan)
                # if use_tfboard:
                    # tf.summary.histogram('conv_encoder', encoder)
        # decoders
        decoder_pan = encoder_pan
        dec_n_ch_list = blocks_list[::-1][1:]
        print(dec_n_ch_list)
        for l_idx, n_ch in enumerate(dec_n_ch_list):
            with K.name_scope('Decoder_block_{0}'.format(l_idx)):
                l_idx_rev = len(blocks_list) - 1 - l_idx
                decoder_pan = concatenate([decoder_pan, list_encoders[l_idx_rev]], axis=concat_axis)
                decoder_pan = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 dilation_rate=(2, 2),
                                 kernel_initializer=k_init)(decoder_pan)
                decoder_pan = AlphaDropout(0.1*l_idx, )(decoder_pan)
                decoder_pan = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(decoder_pan)
                decoder_pan = Conv2DTranspose(filters=n_ch,
                                          kernel_size=k_size,
                                          strides=(2, 2), 
                                          activation=activation,
                                          padding='same',
                                          kernel_initializer=k_init)(decoder_pan)

        # output layer should be softmax
        outp_pan = Conv2DTranspose(filters=n_labels,
                               kernel_size=k_size,
                               activation='sigmoid',
                               padding='same',
                               kernel_initializer='glorot_normal')(decoder_pan)
    
    ### DSM
    
    encoder_dsm = inp_dsm
    
    list_encoders_dsm = []
    
    print('Building Unet for DSM')
    print(blocks_list)   
    
    with K.name_scope('DSM_UNet'):
        for l_idx, n_ch in enumerate(blocks_list):
            with K.name_scope('Encoder_block_{0}'.format(l_idx)):
                encoder_dsm = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(encoder_dsm)
                encoder_dsm = AlphaDropout(0.1*l_idx, )(encoder_dsm)
                encoder_dsm = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 dilation_rate=(2, 2),
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(encoder_dsm)
                list_encoders_dsm.append(encoder_dsm)
                # add maxpooling layer except the last layer
                if l_idx < len(blocks_list) - 1:
                    encoder_dsm = MaxPooling2D(pool_size=(2,2))(encoder_dsm)
                # if use_tfboard:
                    # tf.summary.histogram('conv_encoder', encoder)
        # decoders
        decoder_dsm = encoder_dsm
        dec_n_ch_list = blocks_list[::-1][1:]
        print(dec_n_ch_list)
        for l_idx, n_ch in enumerate(dec_n_ch_list):
            with K.name_scope('Decoder_block_{0}'.format(l_idx)):
                l_idx_rev = len(blocks_list) - 1 - l_idx
                decoder_dsm = concatenate([decoder_dsm, list_encoders[l_idx_rev]], axis=concat_axis)
                decoder_dsm = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 dilation_rate=(2, 2),
                                 kernel_initializer=k_init)(decoder_dsm)
                decoder_dsm = AlphaDropout(0.1*l_idx, )(decoder_dsm)
                decoder_dsm = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(decoder_dsm)
                decoder_dsm = Conv2DTranspose(filters=n_ch,
                                          kernel_size=k_size,
                                          strides=(2, 2), 
                                          activation=activation,
                                          padding='same',
                                          kernel_initializer=k_init)(decoder_dsm)
        
        # output layer should be softmax
        outp_dsm = Conv2DTranspose(filters=n_labels,
                               kernel_size=k_size,
                               activation='sigmoid',
                               padding='same',
                               kernel_initializer='glorot_normal')(decoder_dsm)
        
        outp = concatenate([outp_dsm, outp_pan], axis=concat_axis)
        outp = Conv2D(filters=1, kernel_size=(1,1), padding='same', kernel_initializer='lecun_normal')(outp)

    return Model(inputs=[inp_dsm,inp_pan], outputs=[outp], name=name)

def DiscriminatorNet(inp_DSM, inp_Label, block_list, activation, k_size=(3,3), inputs_ch=64, name='DISCR'):
    
    if K.image_data_format() == 'channels_first':
        concat_axis = 1
    else:
        concat_axis = 3

    k_init = 'lecun_normal'
    with K.name_scope('DiscriminatorNet'):
        with K.name_scope('DSM_input_conv'):
            X = Conv2D(filters=inputs_ch,
                       kernel_size=(1,1),
                       activation=activation,
                       padding='same',
                       kernel_initializer=k_init)(inp_DSM)
        with K.name_scope('Label_input_conv'):  
            Y = Conv2D(filters=inputs_ch,
                       kernel_size=(1,1),
                       activation=activation,
                       padding='same',
                       kernel_initializer=k_init)(inp_Label)
            
        encoder = concatenate([X, Y], axis=concat_axis) 
        for l_idx, n_ch in enumerate(block_list):  #something like [32,32,32,32,32]
            with K.name_scope('Discr_block_{0}'.format(l_idx)):
                encoder = Conv2D(filters=n_ch,
                                 kernel_size=k_size,
                                 activation=activation,
                                 padding='same',
                                 kernel_initializer=k_init)(encoder)
                # encoder = AlphaDropout(0.1*l_idx, )(encoder)
                # add maxpooling layer except the last layer
                if l_idx < len(block_list) - 1:
                    encoder = MaxPooling2D(pool_size=(2,2))(encoder)
        encoder = Flatten()(encoder)
        outp = Dense(1, activation='sigmoid')(encoder)
    
    return Model(inputs=[inp_DSM, inp_Label], outputs=outp, name=name)
