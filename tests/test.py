from binary_factored_pytorch import BinaryFactoredEmbedding
"""
Outline:

BinaryFactoredEmbedding
BinaryFactoredLinear # for output head
BinaryFactoredCrossEntropy

BinaryFactoredEmbedding
 - __init__(num_embeddings: int, kwargs)
 - Testing strategy:
   Partitions: num_embeddings: powers of 2, not powers of 2

BinaryFactoredLinear
 - __init__(num_embeddings: int, embedding_dim: int)
 - Testing strategy:
   Partitions: in_features: powers of 2, not powers of 2

binary_factored_cross_entropy
 - __init__(num_embeddings: int, embedding_dim: int)

 - Testing strategy:
   Partitions: output_logits: torch.Tensor: shape of source_tokens,
               source_tokens: torch.Tensor: tensor of arbitrary shape,
               num_bits: int: > 0
               pad_token: None, >= 0
"""

def test_binary_factored_embedding():
    
