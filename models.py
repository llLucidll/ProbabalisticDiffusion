import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple

#Random seed for reproducibility
random_seed = 6
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VarianceScheduler:
    def __init__(self, beta_start: float=0.0001, beta_end: float=0.02, num_steps: int=1000, interpolation: str='linear') -> None:
        self.num_steps = num_steps

        # find the beta valuess by linearly interpolating from start beta to end beta
        if interpolation == 'linear':
            #: complete the linear interpolation of betas here
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif interpolation == 'quadratic':
            #: complete the quadratic interpolation of betas here
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
        else:
            raise Exception('[!] Error: invalid beta interpolation encountered...')
        

        #: add other statistics such alphas alpha_bars and all the other things you might need here
        self.alphas = 1.0 - self.betas
        #Cumulative product of alphas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        alpha_bars = self.alpha_bars.to(t.device)
        return alpha_bars[t]
    def get_variance(self, t: torch.Tensor) -> torch.Tensor:
        #: get the variance at a given time step
        betas = self.betas.to(t.device)
        return betas[t]

    def add_noise(self, x:torch.Tensor, time_step:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device

        #: sample a random noise
        noise = torch.randn_like(x, device=device)

        betas = self.betas.to(device)
        alpha_bars = self.alpha_bars.to(device)

        #Calculate variance and alpha_bar
        variance = betas[time_step].view(-1, 1, 1, 1)
        alpha_bar = alpha_bars[time_step].view(-1, 1, 1, 1)

        #: construct the noisy sample
        noisy_input = torch.sqrt(alpha_bar) * x + torch.sqrt(1-alpha_bar) * noise

        return noisy_input, noise

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
      super().__init__()
      self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        #: compute the sinusoidal positional encoding of the time
        device = time.device
        

        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        #Apply sin and cos to alternating indices
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


#Conditional Block for injecting time embeddings into Unet Network
class ConditionalBlock(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, in_channels * 2)
        self.in_channels = in_channels

    def forward(self, x, emb):
        # emb: (batch_size, emb_dim)
        emb = self.linear(emb)  # (batch_size, in_channels * 2)
        gamma, beta = emb.chunk(2, dim=1)  # Each: (batch_size, in_channels)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return gamma * x + beta
 
class UNet(nn.Module):
    def __init__(self, in_channels: int=1, 
                 down_channels: List=[64, 128, 128, 128, 128], 
                 up_channels: List=[128, 128, 128, 128, 64], 
                 time_emb_dim: int=128,
                 num_classes: int=10) -> None:
        super().__init__()

        #: You can change the arguments received by the UNet if you want, but keep the num_classes argument
        self.num_classes = num_classes

        #: time embedding layer
        self.time_mlp = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_cond = ConditionalBlock(in_channels = down_channels[0], emb_dim = time_emb_dim)

        #: define the embedding layer to compute embeddings for the labels
        self.class_emb = nn.Embedding(num_classes, time_emb_dim) if num_classes > 0 else None

        # define your network architecture here
        self.downs = nn.ModuleList()
        self.down_cond = nn.ModuleList()
        for i in range(len(down_channels)):
            in_ch = in_channels if i == 0 else down_channels[i-1] # Channels goes to 1, 64, 128, 128, 128
            #Adds a convolutional layer to the downsampling path with the specified number of input and output channels
            self.downs.append(nn.Conv2d(
                in_ch, 
                down_channels[i], 
                kernel_size=3, 
                stride=2, 
                padding=1)
                )
            self.down_cond.append(ConditionalBlock(down_channels[i], time_emb_dim)) 

        #Bottleneck layer:
        #(128, 128, 3, 3)
        self.bottleneck = nn.Conv2d(
            down_channels[-1], 
            down_channels[-1], 
            kernel_size=3, 
            padding=1
            ) 
        self.bottleneck_cond = ConditionalBlock(down_channels[-1], time_emb_dim)

        #Upsampling layer:
        self.ups = nn.ModuleList()
        self.up_cond = nn.ModuleList()
        for i in range(len(up_channels)):
            if i == 0:
                in_ch = down_channels[-1] * 2
            elif i == len(up_channels) - 1:
                in_ch = 192 # 128 + 64 on last skip connection
            else:
                in_ch = up_channels[i-1] * 2
            #in_ch = down_channels[-1] * 2 if i == 0 else (up_channels[i-1] * 2)
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_ch, up_channels[i], kernel_size=3, padding=1)
                )
            )
            self.up_cond.append(ConditionalBlock(up_channels[i], time_emb_dim))

            #Final output layer
        self.output_layer = nn.Conv2d(
            up_channels[-1], 
            in_channels, 
            kernel_size=1
            )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        #: embed time
        t = self.time_mlp(timestep)
        #: handle label embeddings if labels are avaialble
        if label is not None and self.class_emb is not None:
            l = self.class_emb(label)
            t = t + l #Combine time and label embeddings
        
        #: compute the output of your network
        #Downsampling path
        skip_connections = []
        for i, down in enumerate(self.downs):
            x = down(x)
            x = self.down_cond[i](x, t)
            skip_connections.append(x)

        #Bottleneck
        x = self.bottleneck(x)
        x = self.bottleneck_cond(x, t)
        x = F.relu(x)

        #Upsampling path
        for i, up in enumerate(self.ups):
            skip = skip_connections[-(i+1)]
            if x.shape[2:] == skip.shape[2:]:
                x = torch.cat((x, skip), dim=1)
            #print(i)
            x = up(x)
            x = self.up_cond[i](x, t)
            x = F.relu(x)

        #Final output
        out = self.output_layer(x)
        return out

class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int=1, 
                 height: int=32, 
                 width: int=32, 
                 mid_channels: List=[32,32,32], 
                 latent_dim: int=32, 
                 num_classes: int=10) -> None:
        
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # self.mid_size specifies the size of the image [C, H, W] in the bottleneck of the network
        self.mid_size = [mid_channels[2], height // 2**3, width // 2**3] #3 downsampling layers so 2^3
        #encoder_output_dim = self.mid_size[0] * self.mid_size[1] * self.mid_size[2]

        # You can change the arguments of the VAE as you please, but always define self.latent_dim, self.num_classes, self.mid_size
        
        # handle the label embedding here
        self.class_emb = nn.Embedding(num_classes, latent_dim)
        
        # define the encoder part of your network
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels[0], kernel_size=4, stride=2, padding=1), # [1, 32, 32] -> [64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=4, stride=2, padding=1), # [64, 16, 16] -> [128, 8, 8]
            nn.ReLU(),
            nn.Conv2d(mid_channels[1], mid_channels[2], kernel_size=4, stride=2, padding=1), # [128, 8, 8] -> [256, 4, 4]
            nn.ReLU(),
        )

        #Bottleneck layer
        encoder_output_dim = mid_channels[2] * self.mid_size[1] * self.mid_size[2] #8192
        # define the network/layer for estimating the mean
        self.mean_net = nn.Linear(encoder_output_dim, latent_dim)
        # define the networklayer for estimating the log variance
        self.logvar_net = nn.Linear(encoder_output_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + latent_dim, encoder_output_dim) # because of the label embedding mixed in before decode
        # define the decoder part of your network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(mid_channels[2], mid_channels[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[1], mid_channels[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[0], in_channels, kernel_size=4, stride=2, padding=1), # [64, 16, 16] -> [1, 32, 32]
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        
        #compute the output of the network encoder
        out = self.encoder(x) # [B, 256, 4, 4]
        out = out.view(out.size(0), -1)

        #estimating mean and logvar
        mean = self.mean_net(out)
        logvar = self.logvar_net(out)
        
        #computing a sample from the latent distribution
        sample = self.reparameterize(mean, logvar)

        #decoding the sample
        out = self.decode(sample, label)
        return out, mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        #implement the reparameterization trick: sample = noise * std + mean
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        sample = noise * std + mean
        return sample
    
    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        #: compute the binary cross entropy between the pred (reconstructed image) and the traget (ground truth image)
        loss = F.binary_cross_entropy(pred, target, reduction='sum')

        return loss
       
    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        #: compute the KL divergence
        kl_div = -.5 * (logvar.flatten(start_dim=1) + 1 - torch.exp(logvar.flatten(start_dim=1)) - mean.flatten(start_dim=1).pow(2)).sum()

        return kl_div

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            # randomly consider some labels
            labels = torch.randint(0, self.num_classes, [num_samples,], device=device)

        #: sample from standard Normal distrubution
        noise = torch.randn(num_samples, self.latent_dim, device=device)

        #: decode the noise based on the given labels
        out = self.decode(noise, labels)
        return out
    
    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Mixing labels
        label_emb = self.class_emb(labels)  # [B, latent_dim]
        out = torch.cat([sample, label_emb], dim=1)
        out = self.decoder_input(out)
        out = out.view(-1, self.mid_size[0], self.mid_size[1], self.mid_size[2])
        
        #use you decoder to decode a given sample and their corresponding labels
        out = self.decoder(out)
        return out


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor: #Changed from Tuple[torch.Tensor, torch.Tensor]
        #: uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.var_scheduler.num_steps, [x.size(0),], device=x.device)

        #: generate the noisy input
        #variance = self.var_scheduler.get_variance(t).view(-1, 1, 1, 1)
        #alpha_bar = self.var_scheduler.get_alpha_bar(t).view(-1, 1, 1, 1)
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        #: estimate the noise
        estimated_noise = self.network(noisy_input, t, label)

        #: compute the loss (either L1, or L2 loss)
        loss = F.mse_loss(estimated_noise, noise) #L2 LOSS
        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        #: implement the sample recovery strategy of the DDPM
        #Sample recovery strategy : xt-1 = sqrt(1/(alpha)) * (xt - (beta/torch.sqrt(1-alpha_bar) * noise_estimate) + sqrt(beta) * noise

        beta = self.var_scheduler.get_variance(timestep).view(-1, 1, 1, 1)
        alpha = 1.0 - beta
        alpha_bar = self.var_scheduler.get_alpha_bar(timestep).view(-1, 1, 1, 1)


        sqrt_recip_alpha = torch.sqrt(1.0 / alpha)
        sqrt_beta = torch.sqrt(beta)

        sample = sqrt_recip_alpha * (noisy_sample - (beta / torch.sqrt(1 - alpha_bar)) * estimated_noise) + sqrt_beta * torch.randn_like(noisy_sample)
        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None

        #: apply the iterative sample generation of the DDPM
        sample = torch.randn((num_samples, 1,32,32), device = device)
        timesteps = torch.arange(self.var_scheduler.num_steps - 1, -1, -1, device = device) 
        for t in timesteps:
            #Creating a tensor for current time step
            timestep = torch.full((num_samples,), t.item(), device = device, dtype = torch.long)
            #Estimate noise using the network
            estimated_noise = self.network(sample, timestep, labels)
            #Recover the previous sample
            sample = self.recover_sample(sample, estimated_noise, timestep)
        return sample


class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
        #Different seed for DDIM generation 
        random_seed = 7
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.var_scheduler.num_steps, [x.size(0),], device=x.device)

        # generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # estimate the noise
        estimated_noise = self.network(noisy_input, t, label)

        # compute the loss
        loss = F.mse_loss(estimated_noise, noise)

        return loss
    
    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # apply the sample recovery strategy of the DDIM
        alpha_bar_t = self.var_scheduler.get_alpha_bar(timestep)


        #Recover alpha bar from previous sample.
        alpha_bar_prev = self.var_scheduler.get_alpha_bar(timestep - 1)
        alpha_bar_prev = torch.where(timestep > 0, alpha_bar_prev, torch.ones_like(alpha_bar_prev))


         # Compute the square roots needed for the formula
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)        # [B, 1, 1, 1]
        sqrt_alphaie = torch.sqrt(1 - alpha_bar_t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]

        pred_x0 = (noisy_sample - sqrt_alphaie * estimated_noise) / sqrt_alpha_bar_t
        try:
            mean_x_prev = sqrt_alpha_bar_prev * pred_x0 + torch.sqrt(1 - alpha_bar_prev).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * estimated_noise
        except RuntimeError as e:
            print("Error during mean_x_prev computation:")
            print(f"sqrt_alpha_bar_prev: {sqrt_alpha_bar_prev.shape}")
            print(f"pred_x0: {pred_x0.shape}")
            print(f"torch.sqrt(1 - alpha_bar_prev): {torch.sqrt(1 - alpha_bar_prev).shape}")
            print(f"estimated_noise: {estimated_noise.shape}")
            raise e

        #mean_x_prev = sqrt_alpha_bar_prev * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * estimated_noise

        sample = mean_x_prev #Assuming eta is zero, making the process purely deterministic.
        return sample
    
    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None
        # TODO: apply the iterative sample generation of DDIM (similar to DDPM)

        sample = torch.randn((num_samples, 1,32,32), device = device)

        timesteps = torch.arange(self.var_scheduler.num_steps - 1, -1, -1, device = device)

        for t in timesteps:
            timestep = torch.full((num_samples,), t.item(), device = device, dtype = torch.long)

            estimated_noise = self.network(sample, timestep, labels)

            sample = self.recover_sample(sample, estimated_noise, timestep)

        return sample
    
