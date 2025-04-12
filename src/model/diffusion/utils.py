from torch import nn
from linear_attention_transformer import LinearAttentionTransformer

def get_torch_trans(
    heads: int=8, 
    layers: int=1, 
    channels: int=64,
    dim_ff: int=64,
    activation: str='gelu') -> nn.TransformerEncoder:

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

    layer = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
    )

    nn.init.kaiming_normal_(layer.weight)
    return layer
