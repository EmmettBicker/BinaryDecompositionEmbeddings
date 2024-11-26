# flake8: noqa # type: ignore

import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional

def to_binary(
    tensor: torch.Tensor,
    num_bits: int = 18,
    return_type: torch.dtype = torch.float
    ) -> torch.Tensor:
    """Converts an discrete numeric tensor to its binary indexes.
    Turns a tensor of shape (a, b, ..., c) into (a, b, ..., c, num_bits)

    Args:
        tensor (torch.Tensor): A tensor of arbitrary shape.
        num_bits (int, optional): How many binary digits. Defaults to 18.

    Returns:
        torch.Tensor: A tensor of shape cat(tensor.shape, num_bits)
    """
    ...

class BinaryDecompositionEmbedding(nn.Linear):
    r"""Modification of nn.Linear. Pass in a tensor and it gets converted to its
        binary index representation. The binary index is normalized so 0 becomes
        -1 and 1 remains as 1. Then, the binary index representation is passed
        through a linear layer with log2(num_embeddings) inputs and hidden_size output.
         
        Functions as an embedding for tokens by decomposing it into its indices
        and then passing it through a linear layer.
        
        For more information, refer to Pytorch's `nn.Linear` implementation.
        """
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,) -> None: ...
    
    def forward(self, input: Tensor) -> Tensor: ...
    
def binary_decomposition_cross_entropy(
        input: torch.Tensor,
        target: torch.Tensor,
        pad_token: int | None = None,
        reduction: str = "mean",
        ) -> torch.Tensor: