import keras
import numpy as np
import rasterio
from sklearn.utils import shuffle


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, DSM_IDs,
                 PAN_IDs,
                 LABEL_IDs,
                 batch_size=32,
                 shuffle=True,
                 pred_fn=None):
        'Initialization'
        self.DSM_IDs = DSM_IDs
        self.PAN_IDs = PAN_IDs
        self.LABEL_IDs = LABEL_IDs
        self.phase = 'gen'
        self.pred_fn = pred_fn
        if len(self.PAN_IDs) != len(self.DSM_IDs) or len(self.DSM_IDs) != len(self.LABEL_IDs):
            raise ValueError('DSM, PAN or LABEL do not match')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.DSM_IDs) / self.batch_size))

    def getitem(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        DSM_IDs_temp = [self.DSM_IDs[k] for k in indexes]
        PAN_IDs_temp = [self.PAN_IDs[k] for k in indexes]
        LABEL_IDs_temp = [self.LABEL_IDs[k] for k in indexes]

        # Generate data
        DSM, PAN, label = self.__data_generation(DSM_IDs_temp, PAN_IDs_temp, LABEL_IDs_temp)
        
        if self.phase == 'gen':
            y1 = np.ones([label.shape[0], 1])
            return [DSM, PAN, label], [label, y1]

        elif self.phase == 'discr':
        
            pred = self.pred_fn([DSM,PAN])
            
            discr_X_1 = np.concatenate((DSM,DSM), axis=0)
            discr_X_2 = np.concatenate((label,pred), axis=0)
            
            y1 = np.ones(shape=(len(label),1))
            y0 = np.zeros(shape=(len(pred),1))
            
            prob = np.concatenate([y1,y0],axis=0)
            
            #shuffle
            discr_X_1, discr_X2, prob = shuffle(discr_X_1, discr_X_2, prob, random_state=42)
            
                    
            discr_X = [discr_X_1, discr_X_2]
            return discr_X, prob
            

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        DSM_IDs_temp = [self.DSM_IDs[k] for k in indexes]
        PAN_IDs_temp = [self.PAN_IDs[k] for k in indexes]
        LABEL_IDs_temp = [self.LABEL_IDs[k] for k in indexes]

        # Generate data
        DSM, PAN, label = self.__data_generation(DSM_IDs_temp, PAN_IDs_temp, LABEL_IDs_temp)
        
        if self.phase == 'gen':
            y1 = np.ones([label.shape[0], 1])
            return [DSM, PAN, label], [label, y1]

        elif self.phase == 'discr':
        
            pred = self.pred_fn([DSM,PAN])
            
            discr_X_1 = np.concatenate((DSM,DSM), axis=0)
            discr_X_2 = np.concatenate((label,pred), axis=0)
            
            y1 = np.ones(shape=(len(label),1))
            y0 = np.zeros(shape=(len(pred),1))
            
            prob = np.concatenate([y1,y0],axis=0)
            
            #shuffle
            discr_X_1, discr_X2, prob = shuffle(discr_X_1, discr_X_2, prob, random_state=42)
            
                    
            discr_X = [discr_X_1, discr_X_2]
            return discr_X, prob

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.DSM_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, DSM_IDs_temp, PAN_IDs_temp, LABEL_IDs_temp):
        'Generates data containing batch_size samples' 
        # X_out : (n_samples, *dim, n_channels)
        # Y_out : (n_samples, *dim, n_classes)
        # Initialization
        DSM_out = []
        PAN_out = []
        LABEL_out = []
        for i in range(len(DSM_IDs_temp)):
                DSM_out.append(np.moveaxis(rasterio.open(DSM_IDs_temp[i]).read(),0,2))
                PAN_out.append(np.moveaxis(rasterio.open(PAN_IDs_temp[i]).read(),0,2))
                LABEL_out.append(np.moveaxis(rasterio.open(LABEL_IDs_temp[i]).read(),0,2))
       
        return np.asarray(DSM_out), np.asarray(PAN_out), np.asarray(LABEL_out)
