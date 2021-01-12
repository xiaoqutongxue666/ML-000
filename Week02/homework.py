%%cython -a 
cimport numpy as cnp
import numpy as np 
import pandas as pd 
def target_mean_v6(data:pd.DataFrame,y_name:str,x_name:str):
    cdef:
        
        int data_shape=data.shape[0]
        cnp.ndarray[cnp.float64_t]result = np.zeros( data_shape,dtype=np.float64)
        dict value_dict = {}
        dict count_dict = {}
        cnp.ndarray[cnp.int_t]data_x=data[x_name].values
        cnp.ndarray[cnp.int_t]data_y=data[y_name].values 
    for i in range( data_shape):
        data_x_name=data_x[i]
        data_y_name=data_y[i]
        if data_x_name not in value_dict:
            value_dict[data_x_name] = data_y_name
            count_dict[data_x_name] = 1
        else:
            value_dict[data_x_name] += data_y_name
            count_dict[data_x_name] += 1
    for i in range(data_shape):
        data_x_name = data_x[i]
        data_y_name = data_y[i]
        result[i] = (value_dict[data_x_name] - data_y_name) / (count_dict[data_x_name] - 1)
    return result
