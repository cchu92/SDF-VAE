

"""  Collect the 'sdf' and 'iso' value from 
'npy' file, generate train and test data with h5py.
"""

# IMPORT PUBLIC PKGS
import numpy as np
import scipy.ndimage
from sklearn.model_selection import train_test_split
import json



# Read configurations for dataset 
with open('./pre_dataset_config.json') as f:
    config = json.load(f)
path_of_npy = config['path']['path_of_npy']
resol = config['varible']['cell_resolution']# resoluaiton of the sdf
N = config['varible']['Number_of_cell']# number of the cells
test_size = config['varible']['test_size']
index = str(N)
sdfs = np.zeros((N,1,resol,resol,resol), dtype=np.float32)
iso = np.zeros(N)
isovalues = np.zeros(N) 


# read all the 'sdf' and 'iso' value from 'npy' file
for ii in range(N):
    sdf_ = np.load(path_of_npy+str(ii)+'.npy')
    sdf_ = scipy.ndimage.gaussian_filter(sdf_, sigma=0.7).astype(np.float32)
    sdfs[ii, 0, :, :, :]= sdf_
    isovalues[ii] = 0.0 # default isovalue is 0

sdfs_train,sdfs_test,isovalues_train,isovalues_test = train_test_split(sdfs,isovalues,test_size=0.2,random_state=42)

## tricik to keep reduce the  size of dadaset  
# sdfs_train,sdfs_test,isovalues_train,isovalues_test = train_test_split(sdfs_test,iso_test,test_size=0.2,random_state=42)


import h5py
save_train_file = "intermedia_train.h5"
save_test_file = "intermedia_test.h5"

with h5py.File(save_train_file,'w') as f:
    f.create_dataset('sdfs',data =sdfs_train)
    f.create_dataset('isovalues',data =isovalues_train)
with h5py.File(save_test_file,'w') as f:
    f.create_dataset('sdfs',data =sdfs_test)
    f.create_dataset('isovalues',data =isovalues_test)