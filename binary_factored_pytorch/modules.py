import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor


class BinaryDecompositionEmbedding(nn.Linear):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,):
        if not self._is_power_of_two(num_embeddings):
            raise ValueError("num_elements must be a power of two")

        if num_embeddings == 1:
            raise ValueError("num_elements cannot be one")

        self.binary_factors = int(math.log2(num_embeddings))

        super().__init__(  # type: ignore
            self.binary_factors,
            embedding_dim,
            bias,
            device,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        if not self._is_discrete(input):
            raise ValueError(
                "input tensor must be in: "
                "torch.int8, torch.uint8, "
                "torch.int16, torch.int32, "
                "torch.int64"
            )

        binary = to_binary(input, num_bits=self.binary_factors)
        binary_pm = 2 * binary.float() - 1
        return super().forward(binary_pm)

    @staticmethod
    def _is_power_of_two(n: int):
        return n > 0 and int(math.log2(n)) == math.log2(n)

    @staticmethod
    def _is_discrete(t: Tensor):
        return t.dtype in (torch.int8, torch.uint8,
                           torch.int16, torch.int32,
                           torch.int64)


def to_binary(
        tensor: torch.Tensor,
        num_bits: int = 18,
        return_type: torch.dtype = torch.float
        ) -> torch.Tensor:
    binary = tensor.unsqueeze(-1).bitwise_and(
            2**torch.arange(num_bits, dtype=tensor.dtype)
            .__reversed__().to(tensor.device)
        ).bool().to(return_type)
    return binary


def binary_decomposition_cross_entropy(
        input: torch.Tensor,
        target: torch.Tensor,
        pad_token: int | None = None,
        reduction: str = "mean",
        stable_mean: bool = False,
        ) -> torch.Tensor:
    if input.shape[:-1] != target.shape:
        raise ValueError("input.shape must be (*D, num_bits) "
                         "where D = target.shape. Instead recieved "
                         f"input.shape = {input.shape} "
                         f"target.shape = {target.shape}")
    num_bits = input.size(-1)

    if reduction not in ["mean", "none"]:
        raise ValueError("reduction must be 'mean' or 'none'")

    if pad_token is not None:
        padding_mask = (target != pad_token).float()
    else:
        padding_mask = torch.ones_like(target)

    target_binary = to_binary(target, num_bits)

    expanded_mask = padding_mask.unsqueeze(-1).expand_as(target_binary)

    # # Flatten all tensors
    # output_flat = input.reshape(-1)
    # target_flat = target_binary.reshape(-1)
    # mask_flat = expanded_mask.reshape(-1)

    # Compute binary cross entropy with logits
    elementwise_loss = F.binary_cross_entropy_with_logits(
        input.float(),
        target_binary,
        reduction='none'
    )
    # Apply mask and compute mean over non-padding elements
    masked_loss = elementwise_loss * expanded_mask

    if reduction == 'none':
        return masked_loss
    assert reduction == 'mean'
    # elif reduction == 'mean': implicit conditional

    num_valid = expanded_mask.sum()

    # Return average loss over non-padding elements
    return masked_loss.sum() / (num_valid + (1e-6 if stable_mean else 0))
