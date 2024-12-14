import os

import torch.nn as nn

from src.vqvae.decoder import Decoder
from src.vqvae.encoder import Encoder
from src.vqvae.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        h_dim: int = 32,
        res_h_dim: int = 32,
        n_res_layers: int = 2,
        n_embeddings: int = 128,
        embedding_dim: int = 4,
        beta: float = 0.25,
        save_img_embedding_map=False,
    ):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.h_dim = h_dim
        self.n_res_layers = n_res_layers
        self.res_h_dim = res_h_dim
        self.encoder = Encoder(input_dim, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1
        )
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print("original data shape:", x.shape)
            print("encoded data shape:", z_e.shape)
            print("recon data shape:", x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity

    def __str__(self):
        return f"vqvae_{self.h_dim}_{self.res_h_dim}_{self.n_res_layers}"

    def get_save_path(self):
        return os.path.join("checkpoints", "vae", str(self) + ".pt")

    def get_embedding(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        _, z_q, _, _, _ = self.vector_quantization(z_e)
        return z_q
