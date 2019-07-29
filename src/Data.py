import glob
import numpy as np

class Data:
    
    def __init__(self, path, random=False):
        """
        input:
            path: path to the folder with subfolders: DSM, PAN, LABEL
            max_num: int, num of samples
            random: bool, to load samples randomly or from 0 to num_max
        """
        self.DSM = sorted(glob.glob(path+"/DSM/*.tif"))
        self.PAN = sorted(glob.glob(path+"/PAN/*.tif"))
        self.LABEL = sorted(glob.glob(path+"/LABEL/*.tif"))
        if len(self.DSM) != len(self.PAN) or len(self.LABEL) != len(self.PAN):
            raise ValueError('DSM, PAN or LABEL do not match')
      
    def get_data(self, start=0, num=10, as_arr=True, random=False):
        """
        function: load max_num of XY into lists
        output: list of numpy arrays, X (images) and Y (labels)
        """
        DSM_out = []
        PAN_out = []
        LABEL_out = []
      
        if random:
            idx = np.random.choice(list(range(len(self.X))), num, replace=False)
            print('randomly loading {0} tiles from {1} tiles'.format(num, len(self.DSM))) 
        else:
            idx = list(range(start, start+num))
            print('loading {0} - {1} image tiles'.format(start, start+num-1))

        for i in idx:
            DSM_out.append(np.moveaxis(rasterio.open(self.DSM[i]).read(),0,2))
            PAN_out.append(np.moveaxis(rasterio.open(self.PAN[i]).read(),0,2))
            LABEL_out.append(np.moveaxis(rasterio.open(self.LABEL[i]).read(),0,2))
        
        DSM_remove = [self.DSM[i] for i in idx]
        PAN_remove = [self.PAN[i] for i in idx]
        LABEL_remove = [self.LABEL[i] for i in idx]
        
        for i in range(len(DSM_remove)):
            self.DSM.remove(DSM_remove[i])
            self.PAN.remove(PAN_remove[i])
            self.LABEL.remove(LABEL_remove[i])
        
        if as_arr:
            return np.asarray(DSM_out), np.asarray(PAN_out), np.asarray(LABEL_out)
        else:
            return DSM_out, PAN_out, LABEL_out
           
    def split_trn_vld_tst(self, vld_rate=0.2, tst_rate=0.0, random=True, seed=10):
        np.random.seed(seed)

        num = len(self.DSM)
        vld_num = int(num*vld_rate)
        tst_num = int(num*tst_rate)
        
        print('split into {0} train, {1} validation, {2} test samples'.format(num-vld_num-tst_num, vld_num, tst_num))
        idx = np.arange(num)
        if random:
            np.random.shuffle(idx)
        DSM_tst, PAN_tst, LABEL_tst = [self.DSM[k] for k in idx[:tst_num]], [self.PAN[k] for k in idx[:tst_num]], [self.LABEL[k] for k in idx[:tst_num]]
        DSM_vld, PAN_vld, LABEL_vld = [self.DSM[k] for k in idx[tst_num:tst_num+vld_num]], [self.PAN[k] for k in idx[tst_num:tst_num+vld_num]], [self.LABEL[k] for k in idx[tst_num:tst_num+vld_num]]
        DSM_trn, PAN_trn, LABEL_trn = [self.DSM[k] for k in idx[tst_num+vld_num:]], [self.PAN[k] for k in idx[tst_num+vld_num:]], [self.LABEL[k] for k in idx[tst_num+vld_num:]]
        
        
        return DSM_trn, PAN_trn, LABEL_trn, DSM_vld, PAN_vld, LABEL_vld, DSM_tst, PAN_tst, LABEL_tst
