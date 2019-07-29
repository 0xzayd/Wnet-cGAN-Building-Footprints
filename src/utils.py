import numpy as np
from sklearn.preprocessing import MinMaxScaler

def make_trainable(net, val):
    for l in net.layers:
        l.trainable = val

def ary_to_tiles_forward(Gary, Sary, pad=(3,3), shape=(122,122), exclude_empty=False, scale=False): # updated to avoid artefacts

    assert(isinstance(Gary, np.ndarray))
    assert(isinstance(Sary, np.ndarray))
    assert(isinstance(shape, tuple))
    
    ary_height = shape[0] + 2*pad[0]
    ary_width = shape[1] + 2*pad[1]
    ary_list = [] 
    
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
    
    total = 0
    excluded = 0
    sx = -shape[1] - pad[1]
    for _ in range(0, Sary.shape[1]//shape[1]+min(1,(Sary.shape[1]%shape[1]))):
        sx = max(0, sx + ary_width - 2*pad[0])
        ex = sx + ary_width
        sy = -shape[0] - pad[0]
        for _ in range(0, Sary.shape[0]//shape[0]+min(1,(Sary.shape[1]%shape[1]))):           
            sy = max(0, sy + ary_height - 2*pad[1])
            ey = sy + ary_height
            
            crop_ary = Gary[sy:ey,sx:ex]
            
            if scale:
                ascolumns = crop_ary.reshape(-1, 1)
                t = scaler.fit_transform(ascolumns)
                crop_ary = t.reshape(crop_ary.shape)
            
            ary_list.append(crop_ary)
    #if excluded > 0:
        #print("INFO: {0}/{1} tiles were excluded due to not fitting shape {2}".format(excluded, total, shape))
    return np.stack(ary_list), excluded


def tiles_to_ary_forward(stacked_ary, pad=(3,3), Gary_shape_padded=(10986, 10986)):
    
    assert(len(stacked_ary.shape) == 4)
    
    output_ary_height = Gary_shape_padded[0] - 2*pad[0]
    output_ary_width = Gary_shape_padded[1] - 2*pad[1]
    
    padded_tile_height = stacked_ary.shape[1]
    padded_tile_width = stacked_ary.shape[2]
    
    output_ary = np.zeros(shape=(output_ary_height,output_ary_width)+(stacked_ary.shape[3],))    
    output_content = stacked_ary[:,pad[1]:-pad[1],pad[0]:-pad[0],:]
   
    
    index = 0
    for x_step in range(0, output_ary_width, padded_tile_width-2*pad[1]):
        for y_step in range(0, output_ary_height, padded_tile_height-2*pad[0]):
            x0, x1 = x_step, x_step+padded_tile_width-2*pad[1]
            y0, y1 = y_step, y_step+padded_tile_height-2*pad[0]
            
            output_ary[y0:y1, x0:x1] = output_content[index]
            index += 1
    
    return output_ary
    
