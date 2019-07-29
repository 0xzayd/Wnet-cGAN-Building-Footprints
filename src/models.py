import os

from src.networks import Wnet, DiscriminatorNet
from src.utils import make_trainable

import keras

from keras.optimizers import adam
from keras.layers import Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import Input, AveragePooling2D, MaxPooling2D, Dropout, Lambda, AlphaDropout
from keras.layers.merge import concatenate
from keras import backend as K
from keras.models import Model

from keras.callbacks import ModelCheckpoint

import tensorflow as tf
from keras.callbacks import TensorBoard


class Wnet_cgan:
    def __init__(self,
                 height, 
                 width,
                 n_labels=1):
        
        if K.image_data_format() == 'channels_first':
            input_shape = (1, height, width) # 1 because PAN and DSM have only one channel
            concat_axis = 1
        else:
            input_shape = (height, width, 1)
            concat_axis = 3
            
        self.pan_shape = self.dsm_shape = self.label_shape = input_shape
        self.init_epoch = 0
        self.n_labels = n_labels
        
    def build_wnet_cgan(self,
                        wnet_block_list,
                        wnet_k_size, 
                        wnet_activation='selu',
                        wnet_lr=1e-4,
                        discr_inp_channels = 64,
                        discr_block_list=[32,32,32,32,32],
                        discr_k_size=(3,3), 
                        discr_activation='relu',
                        discr_lr=1e-4,
                        lambda_=1e-1):
        inp_dsm = Input(self.dsm_shape, name='dsm_input')
        inp_pan = Input(self.pan_shape, name='pan_input')
        inp_label = Input(self.label_shape, name='label_input')

        wnet_opt = adam(lr=wnet_lr)
        discr_opt = adam(lr=discr_lr)

        # build the Discriminator
        print('Build discr')
        self.discriminator = DiscriminatorNet(inp_dsm,
                                              inp_label,
                                              discr_block_list,
                                              discr_activation,
                                              discr_k_size,
                                              discr_inp_channels,
                                              'Discriminator')
        print('Discriminator Built')
        # make Discriminator untrainable and copy it to 'frozen Discriminator'
        make_trainable(self.discriminator, False)

        frozen_discriminator = Model(inputs=self.discriminator.inputs,
                                     outputs=self.discriminator.outputs,
                                     name='frozen_discriminator')
        frozen_discriminator.compile(discr_opt,
                                     loss = 'binary_crossentropy',
                                     metrics=['accuracy'])
        print('Frozen discriminator and compiled')
        # build the wnet
        print('Build Wnet')
        self.wnet = Wnet(inp_dsm, 
                         inp_pan, 
                         wnet_block_list, 
                         wnet_k_size, 
                         wnet_activation, 
                         self.n_labels, 
                         name='Wnet')
      
        #compile the wnet
        self.wnet.compile(wnet_opt,
                          loss = 'binary_crossentropy',
                          metrics=['accuracy'])  # maybe change to mIoU?

        print('Wnet Built and Compiled') 
        #get the wnet prediction
        pred = self.wnet([inp_dsm, inp_pan])
        # input the prediction into the frozen discriminator and get the probability of realness
        prob = frozen_discriminator([inp_dsm, pred])
        # stack wnet and discriminator to form the Wnet-CGAN
        print('stacking the two')
        self.wnet_cgan = Model(inputs=[inp_dsm, inp_pan, inp_label],
                               outputs=[pred, prob],
                               name='WNet-CGAN')
        print('stacked')
        # compile it
        print('compiling the stcaked')
        self.wnet_cgan.compile(wnet_opt,
                               loss=['binary_crossentropy', 'binary_crossentropy'],
                               loss_weights=[1., lambda_],
                               metrics=['accuracy'])
        print('compiled')
        #print(wnet_cgan.summary())

        # compile the discriminator
        make_trainable(self.discriminator, True)
        self.discriminator.compile(discr_opt,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])

        #print(self.discriminator.summary())
     
            
    def fit_wnet_cgan(self,
                      train_generator,
                      valid_generator,
                      adv_epochs=10,
                      adv_steps_epoch=100,
                      gen_epochs=20,
                      gen_steps_epoch=100,
                      validation_steps=4,
                      n_rounds=10):

        discr_callbacks = self.build_callbacks(monitor='val_acc', phase='discr')
        gen_callbacks = self.build_callbacks(monitor='val_acc', phase='gen')
              
        for i in range(n_rounds):
            #Train Generator first
            train_generator.phase='gen'
            valid_generator.phase='gen'
            self.wnet_cgan.fit_generator(generator=train_generator,
                               validation_data=valid_generator,
                               epochs=(i+1)*gen_epochs,
                               callbacks=gen_callbacks,
                               validation_steps=validation_steps,
                               shuffle=True,
                               steps_per_epoch=gen_steps_epoch,
                               initial_epoch=i*gen_epochs,
                               verbose=1)
            
            self.wnet._make_predict_function()
            train_generator.pred_fn = valid_generator.pred_fn = self.wnet.predict
            train_generator.phase = valid_generator.phase = 'discr'
            
            # train discriminator last
            self.discriminator.fit_generator(generator=train_generator, 
                                   validation_data=valid_generator,
                                   epochs=(i+1)*adv_epochs,
                                   callbacks=discr_callbacks,
                                   validation_steps=validation_steps,
                                   shuffle=True,
                                   steps_per_epoch=adv_steps_epoch,
                                   initial_epoch=i*adv_epochs,
                                   verbose=0)
            
    def build_callbacks(self, use_tfboard=True, monitor=None, phase=None, save=False):
        
        if phase == 'gen':
            path = './results/gen'
        elif phase == 'discr':
            path = './results/discr'

        # Model Checkpoints
        if monitor is None:
            callbackList = []
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            filepath=path+'/weights-{epoch:02d}.hdf5'
            checkpoint = ModelCheckpoint(filepath,
                                         monitor=monitor,
                                         verbose=1,
                                         save_best_only=save,
                                         save_weights_only=True,
                                         mode='max')

            # Bring all the callbacks together into a python list
            callbackList = [checkpoint]
                    
        # Tensorboard
        if use_tfboard:
            if phase is None:
                tfpath = './logs'
            else:
                tfpath = './logs/{0}'.format(phase)
            tensorboard = TrainValTensorBoard(log_dir=tfpath)
            callbackList.append(tensorboard)
        return callbackList
        
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', hist_freq=0, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, histogram_freq=hist_freq, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
