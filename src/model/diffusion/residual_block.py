from torch import nn
from .utils import *

import numpy as np
import torch

class ResidualBlock(nn.Module):
    def __init__(
        self,
        side_dim: int,
        channels: int,
        diffusion_embedding_dim: int,
        nheads: int,
        is_linear: bool=False) -> None:

        super().__init__()
        self.diffusion_projection = nn.Linear(
            in_features=diffusion_embedding_dim,
            out_features=channels,
        )
        self.cond_projection = Conv1d_with_init(
            in_channels=side_dim, 
            out_channels=2*channels,
            kernel_size=1,
        )
        self.mid_projection = Conv1d_with_init(
            in_channels=channels,
            out_channels=2*channels,
            kernel_size=1,
        )
        self.output_projection = Conv1d_with_init(
            in_channels=channels,
            out_channels=2*channels,
            kernel_size=1,
        )

        self.is_linear = is_linear
        if self.is_linear:
            self.time_layer = get_linear_trans(
                heads=nheads,
                layers=1,
                channels=channels,
            )
            self.feature_layer = get_linear_trans(
                heads=nheads,
                layers=1,
                channels=channels,
            )
        else:
            self.time_layer = get_torch_trans(
                heads=nheads,
                layers=1,
                channels=channels,
            )
            self.feature_layer = get_torch_trans(
                heads=nheads,
                layers=1,
                channels=channels,
            )
        
    def forward_time(
        self, 
        x: torch.Tensor, 
        base_shape: list) -> torch.Tensor:
        """
        Forwards data through time dimension
        Args:
            x (torch.Tensor): batch
            base_shape (list): shape of ground truth batch
        Returns:
            x (torch.Tensor): forwarded batch
        """
        B, C, K, L = base_shape

        if L == 1:
            return x

        x = x.reshape(*base_shape).transpose(1, 2)
        x = x.reshape(B*K, C, L)

        if self.is_linear:
            x = self.time_layer(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.time_layer(x.permute(2, 0, 1)).permute(1, 2, 0)
        
        x = x.reshape(B, K, C, L).transpose(1, 2).reshape(B, C, K*L)
        return x 

    def forward_feature(
        self,
        x: torch.Tensor,
        base_shape: list) -> torch.Tensor:
        """
        Forwards data through feature dimension
        Args:
            x (torch.Tensor): batch
            base_shape (list): shape of ground truth batch
        Returns:
            x (torch.Tensor): forwarded batch
        """
        B, C, K, L = base_shape

        if K == 1:
            return x

        x = x.reshape(*base_shape).permute(0, 3, 1, 2)
        x = x.reshape(B*L, C, K)

        if self.is_linear:
            x = self.feature_layer(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.feature_layer(x.permute(2, 0, 1)).permute(1, 2, 0)
        
        x = x.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K*L)
        return x

    def forward(
        self,
        x: torch.Tensor,
        cond_info: torch.Tensor,
        diffusion_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forwards data
        Args:
            x (torch.Tensor): batch
            cond_info (torch.Tensor): accumulated side information
            diffusion_embedding (torch.Tensor): embedding of diffusion step
        Returns:
            x (torch.Tensor): forwarded batch
        """
        B, C, K, L = base_shape = x.shape
        x = x.reshape(B, C, K*L)

        diffusion_embedding = self.diffusion_projection(diffusion_embedding)
        diffusion_embedding = diffusion_embedding.unsqueeze(-1)
        y = x + diffusion_embedding

        y = self.forward_time(x=y, base_shape=base_shape)
        y = self.forward_feature(x=y, base_shape=base_shape)
        y = self.mid_projection(y) # B, 2*C, K*L

        B, CD, K, L = cond_info.shape
        cond_info = cond_info.reshape(B, CD, K*L)
        cond_info = self.cond_projection(cond_info) # B, 2*C, K*L
        y = y + cond_info

        gate, filter_ = torch.chunk(
            input=y,
            chunks=2,
            dim=1,
        )
        y = torch.sigmoid(gate) * torch.tanh(filter_)
        y = self.output_projection(y)

        residual, skip = torch.chunk(
            input=y,
            chunks=2,
            dim=1,
        )
        x = x.reshape(base_shape)
        residual = residual.reshape(*base_shape)
        skip = skip.reshape(*base_shape)

        return (x + residual) / np.sqrt(2.0), skip

