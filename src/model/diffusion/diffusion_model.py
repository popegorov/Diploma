from .diff_embeddings import DiffusionEmbedding
from .residual_block import ResidualBlock
from .utils import Conv1d_with_init

from torch import nn
import math
import torch
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(
        self,
        layers: int,
        channels: int,
        num_steps: int,
        nheads: int,
        diffusion_embedding_dim: int,
        is_linear: bool,
        side_dim: int,
        input_dim: int=2) -> None:

        super().__init__()

        # assert side_dim >= 0, "Incorrect"
        self.channels = channels

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=num_steps,
            embedding_dim=diffusion_embedding_dim,
        )

        self.input_projection = Conv1d_with_init(
            in_channels=input_dim,
            out_channels=self.channels,
            kernel_size=1,
        )
        self.output_projection1 = Conv1d_with_init(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
        )
        self.output_projection2 = nn.Conv1d(
            in_channels=self.channels,
            out_channels=1,
            kernel_size=1,
        )
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=diffusion_embedding_dim,
                    nheads=nheads,
                    is_linear=is_linear,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_info: torch.Tensor,
        diffusion_step: torch.Tensor) -> torch.Tensor:
        """
        Forwards data
        Args:
            x (torch.Tensor): batch
            cond_info (torch.Tensor): accumulated side information
            diffusion_step (torch.Tensor): number of diffusion step
        Returns:
            x (torch.Tensor): forwarded batch through all residual blocks
        """
        B, C, K, L = x.shape
        x = x.reshape(B, C, K*L)
        x = F.relu(self.input_projection(x))
        x = x.reshape(B, self.channels, K, L)

        diffusion_embedding = self.diffusion_embedding(diffusion_step)
        skips = []

        for layer in self.residual_layers:
            x, skip = layer(x, cond_info, diffusion_embedding)
            skips.append(skip)
        
        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.residual_layers)) # division for stability
        x = x.reshape(B, self.channels, K*L)
        x = F.relu(self.output_projection1(x))
        x = self.output_projection2(x).reshape(B, K, L)
        return x
