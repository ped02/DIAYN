import torch

from DIAYN import VAE

class VaeDiscriminator(torch.nn.Module):
    def __init__(self, vae: VAE, discriminator: torch.nn.Module):
        super(VaeDiscriminator, self).__init__()
        self.vae = vae
        self.discriminator = discriminator

    def forward(self, obs: torch.tensor):
        # Get the latent space from the VAE
        mu, log_var = self.vae.encode(obs)
        z = self.vae.reparameterize(mu, log_var)

        # Pass through the discriminator
        output = self.discriminator(z)

        return output