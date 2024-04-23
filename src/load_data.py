'''
custom organize data set for sdfvae analysis
08/03/2024
chenchen.chu@itwm.fraunhofer.de
'''

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


# the size of data N*C*D*D*D, 
# N number of sdf samples
# C channel
# D dimension of each sdf

def custom_transform(sample):
    ''' define the transform of the data from [-1,1] 
    '''
    # Convert numpy array to tensor
    tensor_sample = torch.from_numpy(sample).float()  
    
    # Normalize each sample independently to [-1, 1]
    x_min = tensor_sample.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0].min(dim=-3, keepdim=True)[0]
    x_max = tensor_sample.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0]
    data_tensor_normalized = 2 * ((tensor_sample - x_min) / (x_max - x_min)) - 1

    return data_tensor_normalized 

class custom_datasets(Dataset):
    """Custom dataset for organizing and transforming datasets for PyTorch.

    This dataset class is designed to load data from a specified file path, optionally apply a transformation to the data,
    and support flattening the data for use with fully connected neural network layers.

    Attributes:
        data (numpy.ndarray): The dataset loaded from the specified path, limited to the first two channels.
        channels (int): The number of channels in the dataset.
        dim (int): The dimension of the images (assumed square).
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        flatten (bool): Whether to flatten the data for use with fully connected layers.

    Args:
        data_path (str): The file path to the dataset, expected to be a .npy file.
        transform (callable, optional): Optional transform to apply to each sample.
        flatten (bool): If True, flattens the data for use with fully connected layers. Default is False.
    """

    def __init__(self, data_path, transform=None,flatten = False):
        '''
        Args: loading the dataset 
            data_path: path to the data, and  size should be [N*C*D*D*D]
                N: number of images
                C: number of channels (gray scale = 1, RGB = 3,for this sdf =1)
                D: dimsion of the sdf, D^3
            transform: transform the data, default is None
            flatten:  flatten is used  only for a flatten NN layer, default is False
        '''
        if not data_path:
            raise ValueError("Please provide a valid data_path to your dataset.")
        
        self.data = np.load(data_path)
        self.channels = self.data.shape[1] #number of channels
        # size of each image, square image
        self.dim = self.data.shape[2] 
        self.transform = transform
        self.flatten = flatten
    def __len__(self):
        '''
        Args: return the size of data
        this function will  used for torch.utils.data.DataLoader
        '''
        return len(self.data)
    def __getitem__(self, idx):
        '''
        Thix function will  used for torch.utils.data.DataLoader
        Args: 
            idx: a list, size of batch_size, random choose the index of the data
        '''
        # note, the first dimension is the channel, then height*width
        sample = self.data[idx]
        # label = ... # no label for this dataset


        if self.transform: # if transform is not None, normlize the data
            sample = self.transform(sample)
        if self.flatten:
             sample = sample.view(-1)# when flatten is used for a flatten NN layer
        return sample,idx
    

# # test the data set
# def test_data_load():
#     import torch 
#     from torchvision import transforms
#     from torch.utils.data import DataLoader    
#     data_path = './data/sdf_.npy'
#     load_data = custom_datasets(data_path,transform=custom_transform)
#     # load_data = load_data.to(device)
#     print('dataset size',load_data[0].shape)
#     return 

# test_data_load()

