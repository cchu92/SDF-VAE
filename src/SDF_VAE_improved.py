import torch
import torch.nn as nn
import torch.nn.functional as F


"""
VAE with sdf(3d tensor) and isoavlue(1d) as input to reconstruct the cresspoding sdf and iso
chenchen.chu@itwm.fraunhofer.de
"""


class Encoder(nn.Module):
    def __init__(self, input_channels: int,
                        latent_dim: int, 
                        D: int,
                        hidden_layers: list = None):
        super(Encoder, self).__init__()

        # ======== decoder for sdf==========
        self.hidden_layers = hidden_layers
        
        if hidden_layers is None:
            hidden_layers = [16,32,32, 64, 128]  # Define default layers configuration
        modules = []
        for h_dim in hidden_layers:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(input_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
                )
            input_channels = h_dim
        self.decoder_sdf = nn.Sequential(*modules)
        # ==========decoder for iso value=============
        hidden_layers_iso = [16,32]
        self.hidden_layers_iso = hidden_layers_iso
        modules = []
        input = 1 # for iso vlaue is 1 value
        for h_dim in hidden_layers_iso:
            modules.append(
                nn.Sequential(
                    nn.Linear(input, h_dim),
                    nn.ReLU()))
            input = h_dim   
        self.decoder_iso = nn.Sequential(*modules)
        # ==========latent space layer=============
        self.fc_mu = nn.Linear(hidden_layers[-1]+hidden_layers_iso[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[-1]+hidden_layers_iso[-1], latent_dim)

   
    def reparameterize(self, mu, log_var):
        ''' reparameterise the latent space vector
        mu: mean of latent space vector
        logvar: log variance of latent space vector
        z = mu + std * eps
        '''
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu          


    
    
    def forward(self, sdf,iso):
        ''' Encoder strucure 
        Arg: 
        sdf: torch.tensor 
        iso: torch.tensor
        Return:
        mu: mean
        log_var: std, noise to mean
        z: latent vector
        '''
        iso = torch.unsqueeze(iso, dim=1)# iso size [B] transform into [B,1]
        sdf = self.decoder_sdf(sdf)
        print('shape of sdf', sdf.shape)

        sdf = torch.flatten(sdf, start_dim=1)# [B,N] 
        iso = self.decoder_iso(iso)
        print('shape of iso',iso.shape)

        x_flat = torch.cat([sdf, iso], dim=1)
        print('x_flat shape',x_flat.shape)
        mu = self.fc_mu(x_flat)
        log_var = self.fc_logvar(x_flat)
        z = self.reparameterize(mu, log_var)

        

        return z,mu,log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim: int,
                        D: int,
                        input_channels: int,
                        hidden_layers: list = None):
        super(Decoder, self).__init__()

        if hidden_layers is None:
            hidden_layers=[64,32]
        self.hidden_layers = hidden_layers
        z_dim  = latent_dim
        self.zd_iso = 2
        self.zd_sdf = z_dim - self.zd_iso
    # ======== decoder for sdf==========
        self.N_  = 2
        self.decoder_sdf_first_layer = nn.Linear(self.zd_sdf, hidden_layers[0]*(self.N_)**3 )
        modules = []
        firt_hidden = hidden_layers[0]

        for h_dim in hidden_layers[1:]:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(firt_hidden, h_dim, kernel_size=4, stride=3, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
                )
            firt_hidden = h_dim
        self.decoder_sdf = nn.Sequential(*modules)
        adjt_kernel= 22+19
        # wit 'adjt_kernel' the layer is used to have the recontrcut sdf same dimesion to the original data
        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(hidden_layers[-1],hidden_layers[-1],kernel_size = adjt_kernel,stride=3,padding=1),
            nn.BatchNorm3d(hidden_layers[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_layers[-1], 1, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid())
        self.decoder_iso = nn.Sequential(nn.Linear(self.zd_iso, 20),
                                         nn.ReLU(),
                                         nn.Linear(20, 8),
                                         nn.ReLU(),
                                         nn.Linear(8, 1),
                                         nn.Sigmoid())
    # ======== decoder for iso value=============
    

    def forward(self,z):
        '''
        Decoder strucure
        '''
        # split z into two 2 parts for decoder sdf and iso 
        z_sdf = z[:, :self.zd_sdf]
        z_iso = z[:, -self.zd_iso:]
        sdf_decoded = self.decoder_sdf_first_layer(z_sdf)
        # print('decodeed sdf',sdf_decoded.shape)

        sdf_decoded = sdf_decoded.view(-1, self.hidden_layers[0], self.N_,self.N_,self.N_)
        sdf_decoded = self.decoder_sdf(sdf_decoded)
        sdf_decoded = self.final_layer(sdf_decoded)
        iso_decoded = self.decoder_iso(z_iso).squeeze(-1)
       
        print('decodeed sdf',sdf_decoded.shape)
        print('decodeed iso', iso_decoded.shape)

        return sdf_decoded,iso_decoded

      

class SDF_VAE(nn.Module):
    def __init__(self, 
                 input_channels:int,
                 latent_dim:int,
                 D:int,
                 hidden_layers:list=None):
        super(SDF_VAE, self).__init__()
        

        # 1. ==========Encoder part============
        self.encoder = Encoder(input_channels = input_channels,
                               latent_dim = latent_dim,
                               D = D)
        self.decoder = Decoder(latent_dim = latent_dim,
                               D = D,
                               input_channels = input_channels)


    def forward(self, sdf,iso):
        ''' Ouputs of Deocder and Encoders
            Arg:
            sdf: input 'sdf' 
            iso: input 'isovalue' 
            Return:
            sdf_decoded: reconstruct  'sdf' 
            iso_decoded: reconstruct  isovalue
            mu: mean
            log_var: std, noise to mean
        '''
        z,mu,log_var = self.encoder(sdf, iso)
        sdf_decoded, iso_decoded = self.decoder(z)

        return sdf_decoded, iso_decoded, mu, log_var