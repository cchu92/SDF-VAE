import h5py
import torch
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader


class SDFDataset(Dataset):
    def __init__(self, hdf5_file):
        # Loading the SDF data and isovalues directly here could be inefficient if they are large.
        # Instead, just open and store the file handle, or path, and load individual items in __getitem__.
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, 'r') as file:
            self.length = file['sdfs'].shape[0]
    
    def __len__(self):
        return self.length
    
    def normalize_tensor(self, x,iso):
        # Normalize tensor to range [-1, 1]
        x_min = x.min()
        x_max = x.max()
        x_norm =  ((x - x_min) / (x_max - x_min)) 
        iso_norm =  ((iso - x_min) / (x_max - x_min)) 
        return x_norm,iso_norm
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as file:
            #  SDFs are stored under 'sdfs' and isovalues under 'isovalues'
            sdf = file['sdfs'][idx]
            isovalue = file['isovalues'][idx]

            # Normalize the SDF
            sdf_tensor = torch.tensor(sdf, dtype=torch.float)
            isovalue_tensor = torch.tensor(isovalue, dtype=torch.float)
            sdf_normalized,isovalue_normalized = self.normalize_tensor(sdf_tensor, isovalue_tensor)

            return sdf_normalized, isovalue_normalized

def test_dataset():
    dataset = SDFDataset('sdf_dataset.h5')
    sdf_normalized, isovalue_normalized = dataset[0]

    # Check the range of values in the retrieved SDF
    sdf_min = sdf_normalized.min().item()
    sdf_max = sdf_normalized.max().item()

    print("Min value in the normalized SDF:", sdf_min)
    print("Max value in the normalized SDF:", sdf_max)
    return 

# test_dataset()