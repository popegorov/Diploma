from torch import nn
from linear_attention_transformer import LinearAttentionTransformer

def get_torch_trans(
    heads: int=8, 
    layers: int=1, 
    channels: int=64,
    dim_ff: int=64,
    activation: str='gelu') -> nn.TransformerEncoder:
    """
        Gets torch transformer encoder for short sequences 
        Args:
            heads (int): Transformer number of heads
            layers (int): Transformer number of layers
            channels (int): Transformer input dimension
            dim_ff (int): Transformer feed forward dimension
            activation (str): Transformer activation type
        Returns:
            encoder (nn.Module): Transformer Encoder
    """

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=dim_ff,
        activation=activation,
    )

    return nn.TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=layers,
    )

def get_linear_trans(
    heads: int=8,
    layers: int=1,
    channels: int=64,
    localheads: int=0,
    localwindow: int=0,
    max_seq_len: int=256) -> LinearAttentionTransformer:
    """
        Gets torch transformer encoder for long sequences 
        Args:
            heads (int): Transformer number of heads
            layers (int): Transformer number of layers
            channels (int): Transformer input dimension
            localheads (int): Transformer localheads
            localwindow (str): Transformer localwindow
            max_seq_len (int): max sequence length
        Returns:
            encoder (nn.Module): Transformer Encoder
    """
    return LinearAttentionTransformer(
        dim=channels,
        depth=layers,
        heads=heads,
        max_seq_len=max_seq_len,
        n_local_attn_heads=localheads,
        local_attn_window_size=localwindow,
    )

def Conv1d_with_init(
    in_channels: int,
    out_channels: int,
    kernel_size: int) -> nn.Conv1d:
    """
        Initializes convolution layer 
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of kernel
        Returns:
            layer (nn.Module): initialized convolution layer
    """
    layer = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
    )

    nn.init.kaiming_normal_(layer.weight)
    return layer
