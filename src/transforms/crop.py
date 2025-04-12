import torch
from torch import nn

class Crop(nn.Module):
    """
    Crops stock's history to target sequence length
    """

    def __init__(self, seq_length):
        self.seq_length = seq_length
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): cropped tensor.
        """
        idx = torch.randint(0, len(x) - self.seq_length, [1])
        return x[idx: idx+self.seq_length].flip(0)

