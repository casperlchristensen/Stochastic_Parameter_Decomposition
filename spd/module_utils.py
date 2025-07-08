import math
from functools import reduce
from typing import Any

import einops
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.nn.init import calculate_gain


def get_nested_module_attr(module: nn.Module, access_string: str) -> Any:
    """Access a specific attribute by its full, path-like name.

    Taken from https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8

    Args:
        module: The module to search through.
        access_string: The full name of the nested attribute to access, with each object separated
            by periods (e.g. "linear1.A").
    """
    names = access_string.split(".")
    try:
        mod = reduce(getattr, names, module)
    except AttributeError as err:
        raise AttributeError(f"{module} does not have nested attribute {access_string}") from err
    return mod


@torch.inference_mode()
def remove_grad_parallel_to_subnetwork_vecs(
    A: Float[Tensor, "d_in C"], A_grad: Float[Tensor, "d_in C"]
) -> None:
    """Modify the gradient by subtracting it's component parallel to the activation.

    This is used to prevent any gradient updates from changing the norm of A. This prevents
    Adam from changing the norm due to Adam's (v/(sqrt(v) + eps)) term not preserving the norm
    of vectors.
    """
    parallel_component = einops.einsum(A_grad, A, "d_in C, d_in C -> C")
    A_grad -= einops.einsum(parallel_component, A, "C, d_in C -> d_in C")


def init_param_(
    param: torch.Tensor,
    fan_val: float,
    mean: float = 0.0,
    nonlinearity: str = "linear",
    generator: torch.Generator | None = None,
) -> None:
    """Fill in param with values sampled from a Kaiming normal distribution.

    Args:
        param: The parameter to initialize
        fan_val: The squared denominator of the std used for the kaiming normal distribution
        mean: The mean of the normal distribution
        nonlinearity: The nonlinearity of the activation function
        generator: The generator to sample from
    """
    gain = calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan_val)
    with torch.no_grad():
        param.normal_(mean, std, generator=generator)


def match_dimensions(tensor1: Tensor, tensor2: Tensor, leading: bool = True) -> Tensor:
    """Unsqueezes until tensor1 has the same number of dimensions as tensor2 and then expands it.

    Args:
        tensor1: The first tensor to match.
        tensor2: The second tensor to match.
        leading: If True, add dimensions to the front; otherwise, add to the back.

    Returns:
        A new tensor with dimensions matched to the larger of the two tensors.
    """
    if leading:
        while tensor1.dim() < tensor2.dim():
            tensor1 = tensor1.unsqueeze(0)
        return tensor1.expand(*tensor2.shape[:-1], -1)
    else:
        while tensor1.dim() < tensor2.dim():
            tensor1 = tensor1.unsqueeze(-1)
        return tensor1.expand(-1, *tensor2.shape[1:])
