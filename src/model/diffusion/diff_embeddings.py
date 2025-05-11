from torch import nn
import torch
import torch.nn.functional as F

class DiffusionEmbedding(nn.Module):
    def _build_embedding(
        self,
        num_steps: int,
        dim: int=64) -> torch.Tensor:
        """
        Builds buffer of diffusion embeddings
        Args:
            num_steps (int): number of possible diffusion steps
            dim (int): dimension of diffusion embedding
        Returns:
            table (torch.Tensor): table of embeddings
        """

        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0**(torch.arange(dim) / (dim-1))
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)

        return table

    def __init__(
        self,
        num_steps: int,
        embedding_dim: int=128,
        projection_dim: int=None) -> None:

        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim

        self.register_buffer(
            name="embedding",
            tensor=self._build_embedding(
                num_steps=num_steps,
                dim=embedding_dim / 2,
                ),
            persistent=False,
        )

        self.projection1 = nn.Linear(
            in_features=embedding_dim,
            out_features=projection_dim,
        )

        self.projection2 = nn.Linear(
            in_features=projection_dim,
            out_features=projection_dim,
        )

    def forward(self, diffusion_step: int) -> torch.Tensor:
        """
        Forwards embedding
        Args:
            diffusion_step (int): diffusion step
        Returns:
            embedding (torch.Tensor): diffusion embedding
        """
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        return F.silu(self.projection2(x))
