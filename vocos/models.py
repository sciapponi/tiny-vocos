from typing import Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm

from vocos.modules import ConvNeXtBlock, ResBlock1, AdaLayerNorm

from micromind.networks.xinet import XiConv
from einops import rearrange

class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C,Lupo Olcelli sono attesi, oltre ai familiari, anch L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class XiVocosBackboneFixedChannels(Backbone):

    def __init__(self, input_channels, dim = 128, num_layers = 8):
        super().__init__()
        self.first_layer = XiConv(c_in=input_channels, c_out=dim, kernel_size=(3,1), pool=1, skip_res=None, skip_tensor_in=False)
        self.net = nn.ModuleList(
            [
                XiConv(c_in=dim, c_out=dim, kernel_size=3, pool=1, skip_res=None, skip_tensor_in=False)
                for _ in range(num_layers)
            ]
        )

    def forward(self, input):
        x = input  # batch x freqs x time
        # x = torch.stack([x for _ in range(3)], dim=1) # create fake channels
        x = x[:,None,:,:] # Add 1 channel to make 2d convs work
        x = rearrange(x, "batch channels freqs time -> batch freqs channels time") # rearrange use frequencies as channels
        # print(x.shape)
        x = self.first_layer(x)
        print(x.shape)
        skip = self.net[0](x)

        for conv_block in self.net[1:]:
            skip = conv_block(skip)


        return skip


class XiVocosBackbone(Backbone):
    def __init__(self, 
                 freqs, 
                 dims = [64, 128, 256]):
        super().__init__()
        self.first_layer = XiConv(c_in=freqs, c_out=dims[0], kernel_size=3, pool=0, skip_res=None, skip_tensor_in=False)

        # Up Convolutions
        up = []
        last_size = dims[0]
        for i in dims[1:]:
            up.append(XiConv(c_in=last_size, c_out=i, kernel_size=3, pool=1, skip_res=None, skip_tensor_in=False))
            print(i)
            last_size = i
        self.up = nn.ModuleList(up)

        # Down Convolutions
        down = []
        dims = dims[::-1]
        last_size = dims[0]
        print(last_size)
        for i in dims[1:]:
            down.append(XiConv(c_in=last_size, c_out=i, kernel_size=3, pool=1, skip_res=None, skip_tensor_in=False))
            last_size = i
        self.down = nn.ModuleList(down)

        # self.gen = Generator_XiNet(input_nc=freqs, output_nc=freqs, latent_size=128)
        # self.last_layer = self.last_layer = BlendChannels()
    
    def forward(self, x):
        # x = torch.stack([x for _ in range(3)], dim=1) # create fake channels
        x = x[:,None,:,:]
        x = rearrange(x, "batch channels freqs time -> batch freqs channels time") # rearrange to convolve on frequencies
        print(x.shape)
        x = self.first_layer(x)
        skip = self.up[0](x)
        for conv_block in self.up[1:]:
            skip = conv_block(skip)
        
        skip = self.down[0](skip)

        for conv_block in self.down[1:]:
            skip = conv_block(skip)
        
        # skip = self.last_layer(skip)

        return skip
               

class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x



class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self, input_channels, dim, num_blocks, layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(nn.Conv1d(input_channels, dim, kernel_size=3, padding=1))
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x