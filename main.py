import torch

from models import VAE
from models import DDPM
from models import DDIM
from models import LDDPM
from models import UNet
from models import VarianceScheduler


def prepare_ddpm() -> DDPM:
    
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'


    in_channels = 1
    down_channels = [64, 128, 128, 128, 128]
    up_channels = [128, 128, 128, 128, 64]
    time_embed_dim = 128
    num_classes = 10

    var_scheduler = VarianceScheduler(beta_start=beta1, beta_end=beta2, num_steps=num_steps, interpolation=interpolation)

    network = UNet(in_channels=in_channels, 
                   down_channels=down_channels, 
                   up_channels=up_channels, 
                   time_emb_dim=time_embed_dim,
                   num_classes=num_classes)
    
    ddpm = DDPM(network=network, var_scheduler=var_scheduler)

    return ddpm

def prepare_ddim() -> DDIM:
  
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'
    device = 'cuda'

 
    in_channels = 1
    down_channels = [64, 128, 128, 128, 128]
    up_channels = [128, 128, 128, 128, 64]
    time_embed_dim = 128
    num_classes = 10


    var_scheduler = VarianceScheduler(beta_start=beta1, beta_end=beta2, num_steps=num_steps, interpolation=interpolation)#.to(device)

 
    network = UNet(in_channels=in_channels, 
                   down_channels=down_channels, 
                   up_channels=up_channels, 
                   time_emb_dim=time_embed_dim,
                   num_classes=num_classes)#.to(device)
    
    ddim = DDIM(network=network, var_scheduler=var_scheduler).to(device)

    return ddim

def prepare_vae() -> VAE:
   
 
    in_channels = 1

    mid_channels = [128, 256, 512, 1024, 2048]
    #mid_channels = [32, 64, 128, 256]
    height = width = 32
    latent_dim = 512
    num_classes = 10

    
    vae = VAE(in_channels=in_channels, 
              height=height, 
              width=width, 
              mid_channels=mid_channels, 
              latent_dim=latent_dim,
              num_classes=num_classes)
    
    return vae

def prepare_lddpm() -> LDDPM:
    
    in_channels = 1
    mid_channels = [64, 128, 256, 512]
    height = width = 32
    latent_dim = 1
    num_classes = 10
    vae = VAE(in_channels=in_channels,
              mid_channels=mid_channels,
              height=height,
              width=width,
              latent_dim=latent_dim,
              num_classes=num_classes)

    vae.load_state_dict(torch.load('checkpoints/VAE.pt'))

    # variance scheduler configs
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'

    ddpm_in_channels = latent_dim
    down_channels = [256, 512, 1024]
    up_channels = [1024, 512, 256]
    time_embed_dim = 128


    var_scheduler = VarianceScheduler(beta_start=beta1, beta_end=beta2, num_steps=num_steps, interpolation=interpolation)

   
    network = UNet(in_channels=ddpm_in_channels, 
                   down_channels=down_channels, 
                   up_channels=up_channels, 
                   time_emb_dim=time_embed_dim)
    
    lddpm = LDDPM(network=network, vae=vae, var_scheduler=var_scheduler)

    return lddpm

