import warnings
import pytest
import torch
import torch.nn as nn
from binary_decomposition_pytorch import (BinaryDecompositionEmbedding,
                                          to_binary,
                                          binary_decomposition_cross_entropy)
import torch.nn.functional as F
torch.manual_seed(0x01134)  # type: ignore
"""
Outline:

to_binary
BinaryDecompositionEmbedding
BinaryDecompositionLinear # for output head
BinaryDecompositionCrossEntropy

to_binary
def to_binary(
    tensor: torch.Tensor,
    num_bits: int = 18,
    return_type: torch.dtype = torch.float
    ) -> torch.Tensor:
- Testing strategy:
    Test the following subdivisions:
    tensor: 0dim, 1dim, 2dim
    num_bits: 0, 32>0, 32, 33, >33 | return_type = int
    num_bits: 0, 64>0, 63, 64, >64 | return_type = long

BinaryDecompositionEmbedding
- __init__(num_embeddings: int, kwargs)
- Testing strategy:
    Partitions: num_embeddings: powers of 2, not powers of 2

BinaryDecompositionLinear
- __init__(num_embeddings: int, embedding_dim: int)
- Testing strategy:
    Partitions: in_features: powers of 2, not powers of 2

binary_decomposition_cross_entropy
- __init__(num_embeddings: int, embedding_dim: int)

- Testing strategy:
  Partitions: output_logits: torch.Tensor: shape of source_tokens,
              source_tokens: torch.Tensor: tensor of arbitrary shape,
              num_bits: int: > 0
              pad_token: None, >= 0,
              pad_token: None, !=None,
              reduction: "mean", "none",
              num_bits: 0, 1, >1
"""


def test_to_binary():
    """
    test following subdivisions:
    - tensor: 0dim, 1dim, 2dim
    - num_bits: 0, 32>x>0, 32, >32 | return_type = int
    - return_type: float, bool
    - tensor: int, long
    """
    pairs = [
        # 0d
        (to_binary(torch.tensor(1, dtype=torch.int), num_bits=2),
            torch.tensor([0, 1], dtype=torch.float)),
        # 1d
        (to_binary(torch.tensor([0, 7], dtype=torch.int), num_bits=4),
            torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.float)),
        # 2d
        (to_binary(torch.tensor([[0, 7], [0, 1]], dtype=torch.int), num_bits=4), # noqa
            torch.tensor([[[0, 0, 0, 0], [0, 1, 1, 1]],
                          [[0, 0, 0, 0], [0, 0, 0, 1]]], dtype=torch.float)),
        # integer overflow
        (to_binary(torch.tensor([0, 2**31-1], dtype=torch.int)*2+1, num_bits=32), # noqa
            torch.tensor([[0] * 31 + [1], [1] * 32], dtype=torch.float)),
        # 0 bits WIP
        (to_binary(torch.tensor(1, dtype=torch.int), num_bits=0),
            torch.tensor([], dtype=torch.float)),
        # 1 bit WIP
        (to_binary(torch.tensor(1, dtype=torch.int), num_bits=1),
            torch.tensor([1], dtype=torch.float)),
        # 33 bits
        (to_binary(torch.tensor([0, 2**31-1], dtype=torch.int)*2+1, num_bits=33), # noqa
            torch.tensor([[0] * 32 + [1], [0] + [1] * 32], dtype=torch.float)),
        # bool return type
        (to_binary(torch.tensor([0, 2**31-1], dtype=torch.int)*2+1,
                   num_bits=33, return_type=torch.bool),
            torch.tensor([[False] * 32 + [True], [False] + [True] * 32],
                         dtype=torch.bool)),
        # long input
        (to_binary(torch.tensor([0, 2**33-1], dtype=torch.long), num_bits=33), # noqa
            torch.tensor([[0] * 33, [1] * 33], dtype=torch.float)),
    ]
    for result, expected in pairs:
        assert torch.all(result == expected), \
            f"expected {expected} got {result}"


def test_binary_decomposition_embedding_init():
    """
    partition: num_embed: 0, 1, non_power of 2 > 1, power of 2 > 1
    """
    # 0
    with pytest.raises(ValueError):
        BinaryDecompositionEmbedding(0, 16)

    # 1
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Initializing zero-element tensors is a no-op")
            BinaryDecompositionEmbedding(1, 16)

    # non power of two
    with pytest.raises(ValueError):
        BinaryDecompositionEmbedding(30, 16)

    # power of two
    bde = BinaryDecompositionEmbedding(32, 16)
    e = nn.Embedding(32, 16)

    assert e.num_embeddings == 2 ** bde.in_features, \
        "Hyperparameter mismatch"
    assert e.embedding_dim == bde.out_features, "Hyperparameter mismatch"


def test_binary_decomposition_embedding_running():
    """
    testing: conservation of shape and representation negation check
    """

    # power of two
    bde = BinaryDecompositionEmbedding(32, 16, bias=False)
    e = nn.Embedding(32, 16)
    t = torch.randint(6, size=(6,))
    assert bde(t).shape == e(t).shape

    # 31 and 0 should be additions of i nverse rows of the embedding
    result_31, result_0 = bde.forward(torch.tensor(31)), bde(torch.tensor(0))
    assert torch.all(result_31 + result_0 == torch.zeros_like(result_31))


def test_binary_decomposition_cross_entropy():
    """
    output_logits: 0d, 1d, 2d,
    source_tokens: 0d, 1d, 2d,
    num_bits: int: > 0
    pad_token: None, >= 0,
    pad_token: None, !=None,
    reduction: "mean", "none",
    num_bits: 0, 1, >1
    """

    # no padding, 0d, 1d, 2d
    triplets = (
                 # 0d
                 (torch.tensor([0, 0], dtype=torch.float),  # input
                  torch.tensor(0),  # target
                  torch.tensor([0, 0], dtype=torch.float)),  # test_target
                 # 1d
                 (torch.tensor([[0, 0, 0], [0, 0.5, 0.5]], dtype=torch.float),
                  torch.tensor([0, 3]),
                  torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.float)),
                 # 2d
                 (torch.tensor([[[0, 0, 0]], [[0, 0.5, 0.5]]],
                               dtype=torch.float),
                  torch.tensor([[0], [3]]),
                  torch.tensor([[[0, 0, 0]], [[0, 1, 1]]], dtype=torch.float)),
               )

    for input, target, test_target in triplets:
        result = binary_decomposition_cross_entropy(input=input,
                                                    target=target,
                                                    reduction="none")
        test_result = F.binary_cross_entropy_with_logits(input,
                                                         test_target,
                                                         reduction="none")
        assert torch.all(result == test_result)

        result = binary_decomposition_cross_entropy(input=input,
                                                    target=target,
                                                    reduction="mean")
        test_result = F.binary_cross_entropy_with_logits(input,
                                                         test_target,
                                                         reduction="mean")
        assert torch.all(result == test_result)

    # 1d pad test
    input = torch.tensor([[0, 0, 0], [0, 0.5, 0.5]], dtype=torch.float)
    target = torch.tensor([0, 3])
    test_target = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.float)
    result = binary_decomposition_cross_entropy(input=input,
                                                target=target,
                                                reduction="none",
                                                pad_token=3)
    test_result = F.binary_cross_entropy_with_logits(input,
                                                     test_target,
                                                     reduction="none")
    assert torch.all(result[0] == test_result[0])
    assert torch.all(result[1] == torch.zeros_like(result))

    # -1d / num_bits = 0
    pairs = (
             (torch.tensor(0, dtype=torch.float),
              torch.tensor([])),
            )
    for input, target in pairs:
        with pytest.raises(ValueError):
            result = binary_decomposition_cross_entropy(input=input,
                                                        target=target,
                                                        reduction="none")


test_binary_decomposition_cross_entropy()
