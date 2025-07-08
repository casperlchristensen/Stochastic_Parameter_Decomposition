import importlib
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from jaxtyping import Float
from pydantic import BaseModel, PositiveFloat
from pydantic.v1.utils import deep_update
from torch import Tensor

from spd.log import logger
from spd.spd_types import ModelPath

T = TypeVar("T", bound=BaseModel)

# Avoid seaborn package installation (sns.color_palette("colorblind").as_hex())
COLOR_PALETTE = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#FBAFE4",
    "#949494",
    "#ECE133",
    "#56B4E9",
]


def get_device() -> str:
    # NOTE: MPS returns NaNs on TMS when run. Avoiding for now.
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int | None) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def load_config(config_path_or_obj: Path | str | T, config_model: type[T]) -> T:
    """Load the config of class `config_model`, either from YAML file or existing config object.

    Args:
        config_path_or_obj (Union[Path, str, `config_model`]): if config object, must be instance
            of `config_model`. If str or Path, this must be the path to a .yaml.
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(config_path_or_obj, Path), (
        f"passed config is of invalid type {type(config_path_or_obj)}"
    )
    assert config_path_or_obj.suffix == ".yaml", (
        f"Config file {config_path_or_obj} must be a YAML file."
    )
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)


BaseModelType = TypeVar("BaseModelType", bound=BaseModel)


def replace_pydantic_model(model: BaseModelType, *updates: dict[str, Any]) -> BaseModelType:
    """Create a new model with (potentially nested) updates in the form of dictionaries.

    Args:
        model: The model to update.
        updates: The zero or more dictionaries of updates that will be applied sequentially.

    Returns:
        A replica of the model with the updates applied.

    Examples:
        >>> class Foo(BaseModel):
        ...     a: int
        ...     b: int
        >>> foo = Foo(a=1, b=2)
        >>> foo2 = replace_pydantic_model(foo, {"a": 3})
        >>> foo2
        Foo(a=3, b=2)
        >>> class Bar(BaseModel):
        ...     foo: Foo
        >>> bar = Bar(foo={"a": 1, "b": 2})
        >>> bar2 = replace_pydantic_model(bar, {"foo": {"a": 3}})
        >>> bar2
        Bar(foo=Foo(a=3, b=2))
    """
    return model.__class__(**deep_update(model.model_dump(), *updates))


def compute_feature_importances(
    batch_size: int,
    n_features: int,
    importance_val: float | None,
    device: str,
) -> Float[Tensor, "batch_size n_features"]:
    # Defines a tensor where the i^th feature has importance importance^i
    if importance_val is None or importance_val == 1.0:
        importance_tensor = torch.ones(batch_size, n_features, device=device)
    else:
        powers = torch.arange(n_features, device=device)
        importances = torch.pow(importance_val, powers)
        importance_tensor = einops.repeat(
            importances, "n_features -> batch_size n_features", batch_size=batch_size
        )
    return importance_tensor


def get_lr_schedule_fn(
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"],
    lr_exponential_halflife: PositiveFloat | None = None,
) -> Callable[[int, int], float]:
    if lr_schedule == "linear":
        return lambda step, steps: 1 - (step / steps)
    elif lr_schedule == "constant":
        return lambda *_: 1.0
    elif lr_schedule == "cosine":
        return lambda step, steps: 1.0 if steps == 1 else np.cos(0.5 * np.pi * step / (steps - 1))
    elif lr_schedule == "exponential":
        assert lr_exponential_halflife is not None  # Should have been caught by model validator
        halflife = lr_exponential_halflife
        gamma = 0.5 ** (1 / halflife)
        logger.info(f"Using exponential LR schedule with halflife {halflife} steps (gamma {gamma})")
        return lambda step, steps: gamma**step
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")


def get_lr_with_warmup(
    step: int,
    steps: int,
    lr: float,
    lr_schedule_fn: Callable[[int, int], float],
    lr_warmup_pct: float,
) -> float:
    warmup_steps = int(steps * lr_warmup_pct)
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr * lr_schedule_fn(step - warmup_steps, steps - warmup_steps)

class PnormScheduler:
    """Scheduler for pnorm value during training."""
    
    def __init__(self, start_value=2.0, end_value=0.5, warmup_fraction=0.5, 
                 schedule_type='linear', total_steps=None):
        """
        Initialize the pnorm scheduler.
        
        Args:
            start_value: Initial pnorm value (default: 2.0)
            end_value: Final pnorm value (default: 0.5)
            warmup_fraction: Fraction of training before scheduling starts (default: 0.5)
            schedule_type: Type of scheduling ('linear', 'cosine', 'exponential')
            total_steps: Total number of training steps (optional, can be set later)
        """
        self.start_value = start_value
        self.end_value = end_value
        self.warmup_fraction = warmup_fraction
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        
    def set_total_steps(self, total_steps):
        """Set total steps if not provided during initialization."""
        self.total_steps = total_steps
        
    def get_pnorm(self, current_step):
        """
        Get the pnorm value for the current training step.
        
        Args:
            current_step: Current training step
            
        Returns:
            float: The pnorm value for this step
        """
        if self.total_steps is None:
            raise ValueError("Total steps must be set before using the scheduler")
            
        # Before warmup period, return start value
        warmup_steps = int(self.warmup_fraction * self.total_steps)
        if current_step < warmup_steps:
            return self.start_value
            
        # Calculate progress after warmup
        remaining_steps = self.total_steps - warmup_steps
        progress = (current_step - warmup_steps) / remaining_steps
        progress = min(1.0, progress)  # Clamp to [0, 1]
        
        # Apply scheduling
        if self.schedule_type == 'linear':
            return self._linear_schedule(progress)
        elif self.schedule_type == 'cosine':
            return self._cosine_schedule(progress)
        elif self.schedule_type == 'exponential':
            return self._exponential_schedule(progress)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
            
    def _linear_schedule(self, progress):
        """Linear interpolation between start and end values."""
        return self.start_value + (self.end_value - self.start_value) * progress
        
    def _cosine_schedule(self, progress):
        """Cosine annealing schedule."""
        import math
        cosine_progress = 0.5 * (1 + math.cos(math.pi * progress))
        return self.end_value + (self.start_value - self.end_value) * cosine_progress
        
    def _exponential_schedule(self, progress):
        """Exponential decay schedule."""
        import math
        # Calculate decay rate
        decay_rate = math.log(self.end_value / self.start_value)
        return self.start_value * math.exp(decay_rate * progress)

def get_pnorm_schedule_fn(
    start_value: float,
    end_value: float,
    warmup_fraction: float,
    schedule_type: Literal["linear", "cosine", "exponential"],
    total_steps: int | None = None,
) -> Callable[[int], float]:
    """Get a function that returns the pnorm value for a given step."""
    return PnormScheduler(start_value, end_value, warmup_fraction, schedule_type, total_steps).get_pnorm


def replace_deprecated_param_names(
    params: dict[str, Float[Tensor, "..."]], name_map: dict[str, str]
) -> dict[str, Float[Tensor, "..."]]:
    """Replace old parameter names with new parameter names in a dictionary.

    Args:
        params: The dictionary of parameters to fix
        name_map: A dictionary mapping old parameter names to new parameter names
    """
    for k in list(params.keys()):
        for old_name, new_name in name_map.items():
            if old_name in k:
                params[k.replace(old_name, new_name)] = params[k]
                del params[k]
    return params


def resolve_class(path: str) -> type[nn.Module]:
    """Load a class from a string indicating its import path.

    Args:
        path: The path to the class, e.g. "transformers.LlamaForCausalLM" or
            "spd.experiments.resid_mlp.models.ResidMLP"
    """
    module_path, _, class_name = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_pretrained(
    path_to_class: str,
    model_path: ModelPath | None = None,
    model_name_hf: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Load a model from a path to the class and a model name or path.

    Loads from either huggingface (if model_name_hf is provided) or from a wandb str or local path
    (if model_path is provided).

    Args:
        path_to_class: The path to the class, e.g. "transformers.LlamaForCausalLM" or
            "spd.experiments.resid_mlp.models.ResidMLP"
        model_path: The path to the model, e.g. "wandb:spd-train-resid-mlp/runs/zas5yjdl" or
            "/path/to/model/checkpoint"
        model_name_hf: The name of the model in the Hugging Face model hub,
            e.g. "SimpleStories/SimpleStories-1.25M"
    """
    assert model_path is not None or model_name_hf is not None, (
        "Either model_path or model_name_hf must be provided."
    )
    model_cls = resolve_class(path_to_class)
    if not hasattr(model_cls, "from_pretrained"):
        raise TypeError(f"{model_cls} lacks a `from_pretrained` method.")
    return model_cls.from_pretrained(model_path or model_name_hf, **kwargs)  # type: ignore


def extract_batch_data(
    batch_item: dict[str, Any] | tuple[torch.Tensor, ...] | torch.Tensor,
    input_key: str = "input_ids",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract input data from various batch formats.

    This utility function handles different batch formats commonly used across the codebase:
    1. Dictionary format: {"input_ids": tensor, ...} - common in LM tasks
    2. Tuple format: (input_tensor, labels) - common in SPD optimization
    3. Direct tensor: when batch is already the input tensor

    Args:
        batch_item: The batch item from a data loader
        input_key: Key to use for dictionary format (default: "input_ids")

    Returns:
        The input tensor extracted from the batch
    """
    if isinstance(batch_item, dict):
        # Dictionary format: extract the specified key
        if input_key not in batch_item:
            available_keys = list(batch_item.keys())
            raise KeyError(
                f"Key '{input_key}' not found in batch. Available keys: {available_keys}"
            )
        tensor = batch_item[input_key]
        label = batch_item.get("labels", None)
    elif isinstance(batch_item, tuple):
        # Assume input is the first element
        tensor = batch_item[0]
        label = batch_item[1] if len(batch_item) > 1 else None
    elif isinstance(batch_item, torch.Tensor):
        # Direct tensor format
        tensor = batch_item
        label = None
    else:
        raise TypeError(f"Unsupported batch format: {type(batch_item)}. ")

    return tensor, label


def calc_kl_divergence_lm(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
) -> Float[Tensor, ""]:
    """Calculate the KL divergence between two logits."""
    assert pred.shape == target.shape
    log_q = torch.log_softmax(pred, dim=-1)  # log Q
    p = torch.softmax(target, dim=-1)  # P
    kl = F.kl_div(log_q, p, reduction="none")  # P · (log P − log Q)
    return kl.sum(dim=-1).mean()  # Σ_vocab / (batch·seq)


def calc_mean_squared_error(
    pred: Float[Tensor, "..."],
    target: Float[Tensor, "..."],
) -> Float[Tensor, ""]:
    """Calculate the mean squared error loss."""
    assert pred.shape == target.shape
    return F.mse_loss(pred, target, reduction="mean")  # Mean over all dimensions
