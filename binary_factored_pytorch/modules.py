import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BinaryFactoredEmbedding(nn.Embedding):  # type: ignore
    def __init__(self, num_embeddings: int, **kwargs):  # type: ignore
        """
        Args:
            num_embeddings (int): _description_
            embedding_dim (int): _description_
        """
        assert math.log2(num_embeddings) == int(math.log2(num_embeddings)), \
            "num_elements must be a power of two"

        super().__init__(  # type: ignore
            int(math.log2(num_embeddings)),
            kwargs
        )



# def to_binary(tensor: torch.Tensor, num_bits: int = 18) -> torch.Tensor:
#     tensor = tensor.to(torch.int32)
#     binary = tensor.unsqueeze(-1).bitwise_and(
#         2**torch.arange(num_bits, dtype=torch.int32).to(tensor.device)
#         ).bool().float()
#     return binary


# def binary_embedding(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
#     # Convert to bits
#     binary = tensor.unsqueeze(-1).bitwise_and(
#         2**torch.arange(num_bits).to(tensor.device)).bool()
#     # binary = binary.flip(-1)

#     # Convert zeros to -1s
#     binary_pm = 2 * binary.float() - 1
#     return self.binary_text_embeddings(binary_pm)


# def masked_binary_cross_entropy(output_logits: torch.Tensor,
#                                 source_tokens: torch.Tensor,
#                                 pad_token: int,
#                                 num_bits: int = 18) -> torch.Tensor:
#     padding_mask = (source_tokens != pad_token).float()
#     target_binary = to_binary(source_tokens, num_bits)

#     expanded_mask = padding_mask.unsqueeze(-1).expand_as(target_binary)

#     # Flatten all tensors
#     output_flat = output_logits.reshape(-1)
#     target_flat = target_binary.reshape(-1)
#     mask_flat = expanded_mask.reshape(-1)

#     # Compute binary cross entropy with logits
#     elementwise_loss = F.binary_cross_entropy(
#         torch.sigmoid(output_flat),
#         target_flat,
#         reduction='none'
#     )

#     # Apply mask and compute mean over non-padding elements
#     masked_loss = elementwise_loss * mask_flat
#     num_valid = mask_flat.sum()

#     # Return average loss over non-padding elements
#     return masked_loss.sum() / (num_valid + 1e-6)
