import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, 
        noise: torch.Tensor, 
        predicted: torch.Tensor,
        observed_mask: torch.Tensor,
        cond_mask: torch.Tensor,
        **batch) -> dict:
        """
        Args:
            predicted (Tensor): model output predicted noise.
            noise (Tensor): ground-truth noise.
            observed_mask (Tensor): indicator mask for observed objects in batch
            cond_mask (Tensor): random mask for training objects in batch

        Returns:
            losses (dict): dict containing calculated loss functions.
        """

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        return {"loss": loss}
