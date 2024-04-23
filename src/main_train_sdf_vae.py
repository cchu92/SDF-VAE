# ========public pkgs========
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import numpy as np
#========= private pkgs==========================
from load_data import custom_datasets
from load_data import custom_transform

from load_data_h5py import SDFDataset
from SDF_VAE_improved import SDF_VAE


# ==================Load configuration file===============
with open('./config_SDF_VAE.json') as f:
    config = json.load(f)

# Extract configuration parameters
batch_size = config['model_params']['batch_size']
latent_dim = config['model_params']['latent_dim']
beta = config['model_params']['beta']
learning_rate = config['train_params']['learning_rate']
epochs = config['train_params']['epochs']
manual_seed = config['random_seed']['manual_seed']
cuda_manual_seed = config['random_seed']['cuda_manual_seed']
loading_checkpoint = config['train_params']['loading_checkpoint']
# Paths from configuration
data_path_train = config['Path']['train_data_path']
data_path_test = config['Path']['test_data_path']
save_path = config['Path']['save_path']
checkpoint_path = config['Path']['log_path']


# Set random seeds for reproducibility
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(cuda_manual_seed)
sdf_dimen = 30
# load test and train data
dataset_train = SDFDataset(data_path_train)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_train = SDFDataset(data_path_test)
loader_test = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
model = SDF_VAE(input_channels=1, latent_dim=latent_dim, D=sdf_dimen)

# ========Setup device (GPU/CPU)
if torch.cuda.is_available(): # GPU is available
    print('GPU is available')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(str(torch.cuda.device_count()),'GPUS are available')
    device = torch.device("cuda:0")
    GPU = True
    model.to(device)
else:  # only cpu is available
    print('CPU is only available')
    GPU = False
    device = torch.device("cpu")
    model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


# Define the loss function
def lossfunc(sdf,sdf_hat,iso,iso_hat,mu,logvar,beta):
    """
    Computes the Variational Autoencoder (VAE) loss function, combining reconstruction loss and KL divergence.
    Args
    Returns:
        torch.Tensor: The computed loss value.
    """
    sdf_loss = F.mse_loss(sdf_hat, sdf,reduction = 'sum')
    iso_loss = F.mse_loss(iso_hat, iso,reduction = 'sum')*40**3
    # iso_loss = F.mse_loss(iso_hat, iso,reduction = 'sum')*30**3 # for another case
    recons_loss = sdf_loss+ iso_loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)
    return recons_loss + beta* kl_loss


# Load checkpoint if specified
if loading_checkpoint:
    checkpoint = torch.load(checkpoint_path+'checkpoint.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else: # use the initial model and optimiser
    current_epoch = 0
    
for epoch in range(current_epoch, epochs+1):
    model.train()

    mu_list = list()
    sdf_test_list = list()
    isov_test_list = list()
    model.train()
    train_loss = 0
    for sdf_data, isovalues in loader_train:
        sdf_data = sdf_data.to(device)
        isovalues = isovalues.to(device)
        sdf_hat,iso_hat,mu,logvar = model(sdf_data,isovalues)
        mu_list.append(mu.detach()) # save latent vector
        #==== forwad pass
        loss = lossfunc(sdf_data,sdf_hat,isovalues,iso_hat,mu,logvar,beta)
        train_loss += loss.item()
        #==== backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #=== save model 
    if (epoch) % 4 == 0: # save model every 40 steps
        mu_list = torch.cat(mu_list,dim=0)
        save_model = 'VAEmodel_'+str(epoch) + '.pt'

        # if GPU:
        #     torch.save(model.module.state_dict(), save_path+save_model) # save module on GPU
        # else:
        torch.save(model.state_dict(), save_path+save_model) # save model on CPU 

        save_mu = 'mu_list_'+str(epoch) + '.pt'
        torch.save(mu_list, save_path+save_mu)
        # ===== checke the loss function on test dataset
        with torch.no_grad():
            model.eval()
            mu_list_test = list()
            test_loss = 0
            for sdf_data, isovalues in loader_test:
                sdf_data = sdf_data.to(device)
                isovalues = isovalues.to(device)
                sdf_hat,iso_hat,mu,logvar = model(sdf_data,isovalues)
                loss = lossfunc(sdf_data,sdf_hat,isovalues,iso_hat,mu,logvar,beta)
                test_loss += loss.item()
                mu_list_test.append(mu.detach())
                sdf_test_list.append(sdf_data.detach())
                isov_test_list.append(isovalues.detach())
            mu_list_test = torch.cat(mu_list_test,dim=0)
            save_mu = 'mu_list_test_'+str(epoch) + '.pt'
            torch.save(mu_list_test, save_path+save_mu)
            
            sdf_test_list = torch.cat(sdf_test_list,dim=0)
            save_sdf_test = 'sdf_test_'+str(epoch) + '.pt'

            iso_test_list = torch.cat(isov_test_list,dim=0)
            iso_sdf_test = 'iso_test_'+str(epoch) + '.pt'
            torch.save(iso_test_list, save_path+iso_sdf_test)
            


    print(f'====> Epoch: {epoch} Average loss:{train_loss/len(loader_train.dataset):.4f}')


    #%% save checkpoint
torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_path+'checkpoint.tar')