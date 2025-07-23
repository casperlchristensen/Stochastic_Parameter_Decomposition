from typing import Any, Callable, Literal, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_stochastic_masks
from spd.models.components import EmbeddingComponent, LinearComponent
from spd.utils import calc_kl_divergence_lm


def calc_embedding_recon_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    component: EmbeddingComponent,
    masks: list[dict[str, Float[Tensor, "... C"]]],
    embed_module_name: str,
    unembed: bool = False,
) -> Float[Tensor, ""]:
    """
    recon loss that directly compares the outputs of the (optionally masked)
    ``EmbeddingComponent``(s) to the outputs of the original ``nn.Embedding`` modules.

    If ``unembed`` is ``True``, both the masked embedding output and the target embedding
    output are unembedded using the ``lm_head`` module, and the KL divergence is used as the loss.

    If ``unembed`` is ``False``, the loss is the MSE between the masked embedding output
    and the target embedding output is used as the loss.
    """

    # --- original embedding output --------------------------------------------------------- #
    orig_module = model.model.get_submodule(embed_module_name)
    assert isinstance(orig_module, nn.Embedding), (
        f"Module {embed_module_name} expected to be nn.Embedding, got {type(orig_module)}"
    )
    target_out: Float[Tensor, "... d_emb"] = orig_module(batch)

    # --- masked embedding output ----------------------------------------------------------- #
    loss = torch.tensor(0.0, device=component.A.device)
    for mask_info in masks:
        component.mask = mask_info[embed_module_name]

        masked_out: Float[Tensor, "... d_emb"] = component(batch)  # type: ignore[arg-type]
        component.mask = None

        if unembed:
            assert hasattr(model.model, "lm_head"), "Only supports unembedding named lm_head"
            target_out_unembed = model.model.lm_head(target_out)
            masked_out_unembed = model.model.lm_head(masked_out)
            loss += calc_kl_divergence_lm(pred=masked_out_unembed, target=target_out_unembed)
        else:
            loss += ((masked_out - target_out) ** 2).sum(dim=-1).mean()

    loss /= len(masks)

    return loss


def calc_schatten_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    pnorm: float,
    components: dict[str, LinearComponent | EmbeddingComponent],
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the Schatten loss on the active components.

    The Schatten loss is calculated as:
        L = Σ_{components} mean(ci_upper_leaky^pnorm · (||A||_2^2 + ||B||_2^2))

    where:
        - ci_upper_leaky are the upper leaky relu causal importances for each component
        - pnorm is the power to raise the mask to
        - A and B are the component matrices
        - ||·||_2 is the L2 norm

    Args:
        ci_upper_leaky: Dictionary of upper leaky relu causal importances for each layer.
        pnorm: The pnorm to use for the importance minimality loss. Must be positive.
        components: Dictionary of components for each layer.
        device: The device to compute the loss on.

    Returns:
        The Schatten loss as a scalar tensor.
    """

    total_loss = torch.tensor(0.0, device=device)
    for component_name, component in components.items():
        A_norms = component.A.square().sum(dim=-2)
        B_norms = component.B.square().sum(dim=-1)
        schatten_norms = A_norms + B_norms
        loss = einops.einsum(
            ci_upper_leaky[component_name] ** pnorm, schatten_norms, "... C, C -> ..."
        )
        total_loss += loss.mean()
    return total_loss


def calc_importance_minimality_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]], pnorm: float
) -> Float[Tensor, ""]:
    """Calculate the importance minimality loss on the upper leaky relu causal importances.

    Args:
        ci_upper_leaky: Dictionary of causal importances upper leaky relu for each layer.
        pnorm: The pnorm to use for the importance minimality loss. Must be positive.

    Returns:
        The importance minimality loss on the upper leaky relu causal importances.
    """
    total_loss = torch.zeros_like(next(iter(ci_upper_leaky.values())))

    for layer_ci_upper_leaky in ci_upper_leaky.values():
        # Note, the paper uses an absolute value but our layer_ci_upper_leaky is already > 0
        total_loss = total_loss + layer_ci_upper_leaky**pnorm

    # Sum over the C dimension and mean over the other dimensions
    return total_loss.sum(dim=-1).mean()


def calc_masked_recon_layerwise_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    device: str,
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: list[dict[str, Float[Tensor, "... C"]]],
    target_out: Float[Tensor, "... d_model_out"],
    loss_type: Literal["mse", "kl", "ce-diff"] = "kl",
) -> Float[Tensor, ""]:
    """Calculate the recon loss when augmenting the model one (masked) component at a time."""
    total_loss = torch.tensor(0.0, device=device)
    for mask_info in masks:
        for component_name, component in components.items():
            modified_out = model.forward_with_components(
                batch,
                components={component_name: component},
                masks={component_name: mask_info[component_name]},
            )
            if loss_type == "mse":
                loss = ((modified_out - target_out) ** 2).mean()
            elif loss_type == "kl":
                loss = calc_kl_divergence_lm(pred=modified_out, target=target_out)
            else:
                raise ValueError(f"Invalid loss type: {loss_type}")
            total_loss += loss
    n_modified_components = len(masks[0])
    return total_loss / (n_modified_components * len(masks))

from typing import Literal, Optional, Union
from dataclasses import dataclass

def calculate_possible_losses(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
    config: Config,
    batch: Optional[Int[Tensor, "..."]] = None,
    target_ce_loss: Optional[Float[Tensor, ""]] = None,
) -> dict[str, Float[Tensor, ""]]:
    """Calculate all requested loss types and return a dictionary"""
    losses = {}
    
    for loss_type in config.output_loss_types:
        if loss_type == "kl":
            losses["kl"] = calc_kl_divergence_lm(pred=pred, target=target)
        elif loss_type == "kl_top_k":
            losses["kl_top_k"] = calc_kl_top_k(
                pred=pred, target=target, k=config.output_loss_types[loss_type].k
            )
        elif loss_type == "ce_labels":
            # Get the ground truth tokens (excluding the first token)
            ground_truth_tokens = batch[:, 1:]  # Shape: (batch_size, seq_len-1)
            
            # Also trim the logits to match - remove the last position since we don't have a target for it
            pred_trimmed = pred[:, :-1, :]      # Shape: (batch_size, seq_len-1, vocab_size)
            target_trimmed = target[:, :-1, :]  # Shape: (batch_size, seq_len-1, vocab_size)
            
            # Gather logits for correct tokens
            pred_correct_logits = torch.gather(pred_trimmed, -1, ground_truth_tokens.unsqueeze(-1)).squeeze(-1)
            target_correct_logits = torch.gather(target_trimmed, -1, ground_truth_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Compute log_softmax efficiently
            pred_lse = torch.logsumexp(pred_trimmed, dim=-1)
            target_lse = torch.logsumexp(target_trimmed, dim=-1)
            
            # Get probabilities for correct tokens
            pred_log_probs_correct = pred_correct_logits - pred_lse
            target_probs_correct = torch.exp(target_correct_logits - target_lse)
            
            # Compute cross-entropy loss
            ce_loss = -pred_log_probs_correct * target_probs_correct.detach()
            losses["ce_labels"] = ce_loss.mean()
    return losses

def calc_kl_top_k(
    pred: Float[Tensor, "... vocab"],
    target: Float[Tensor, "... vocab"],
    k: int = 5
) -> Float[Tensor, ""]:
    #TODO Try different ways of doing this
    target_probs = F.softmax(target, dim=-1)
    top_k_values, top_k_indices = torch.topk(target_probs, k=k, dim=-1)
    
    # Gather the corresponding logits
    target_top_k_logits = torch.gather(target, -1, top_k_indices)
    pred_top_k_logits = torch.gather(pred, -1, top_k_indices)
    
    # Compute cross-entropy over just these k logits
    target_top_k_probs = F.softmax(target_top_k_logits, dim=-1)
    pred_top_k_log_probs = F.log_softmax(pred_top_k_logits, dim=-1)
    
    # Cross-entropy: -sum(p_target * log(p_pred))
    ce = -(target_top_k_probs * pred_top_k_log_probs).sum(dim=-1)
    return ce.mean()

def calc_ce_diff(
    pred: Float[Tensor, "... vocab"],
    batch: Int[Tensor, "..."],
    target_ce_loss: Float[Tensor, ""]
) -> Float[Tensor, ""]:
    """Calculate cross-entropy difference"""
    flat_batch = batch[:, 1:].flatten()
    flat_pred = einops.rearrange(pred[:, :-1], "... vocab -> (...) vocab")
    pred_ce_loss = F.cross_entropy(input=flat_pred, target=flat_batch)
    return pred_ce_loss - target_ce_loss

def calc_reconstruction_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... d_model_out"],
    config: Config,
    target_ce_loss: Optional[Float[Tensor, ""]] = None,
) -> dict[str, Float[Tensor, ""]]:
    """
    Unified reconstruction loss calculation supporting multiple loss types.
    
    Args:
        filler_config: Dictionary with 'use_filler' and 'filler_scalar' keys
    
    Returns:
        Dictionary of loss values by type
    """
    # Forward pass with components
    out = model.forward_with_components(
        batch, 
        components=components, 
        masks=masks,
        filler_comp_scalar=config.filler_scalar if config.learned_filler_comp else 0.0
    )
    
    # Calculate all requested losses
    losses = calculate_possible_losses(
        pred=out,
        target=target_out,
        config=config,
        batch=batch,
        target_ce_loss=target_ce_loss
    )
    
    return losses

def calc_masked_recon_loss(
    model: ComponentModel,
    batch: Float[Tensor, "... d_in"],
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... d_mdoel_out"],
    loss_type: Literal["mse", "kl", "ce-diff"] = "mse",
) -> Float[Tensor, ""]:
    """Calculate the MSE over all masks."""
    # Do a forward pass with all components
    out = model.forward_with_components(batch, components=components, masks=masks)
    if loss_type == "mse":
        loss = ((out - target_out) ** 2).mean()
    elif loss_type == "kl":
        loss = calc_kl_divergence_lm(pred=out, target=target_out)
    elif loss_type == "ce-diff":
        # loss = calc_kl_divergence_lm(pred=out, target=target_out)

        flat_batch = batch[:, 1:].flatten()
        flat_out = einops.rearrange(out[:, :-1], "... vocab -> (...) vocab")
        flat_target_out = einops.rearrange(target_out[:, :-1], "... vocab -> (...) vocab")
        original_ce_loss = F.cross_entropy(input=flat_target_out, target=flat_batch)
        component_ce_loss = F.cross_entropy(input=flat_out, target=flat_batch)
        loss = component_ce_loss - original_ce_loss
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss


def _calc_tensors_mse(
    params1: dict[str, Float[Tensor, "d_in d_out"]],
    params2: dict[str, Float[Tensor, "d_in d_out"]],
    scalers: dict[str, Float[Tensor, "1"] | Float[Tensor, "d_in d_out"]],
    n_params: int,
    device: str,
) -> Tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    """Calculate the MSE between params1 and params2, summing over the d_in and d_out dimensions.

    Normalizes by the number of parameters in the model.

    Args:
        params1: The first set of parameters
        params2: The second set of parameters
        scalers: A dictionary of scalers for each parameter, used to weight the loss
        n_params: The number of parameters in the model
        device: The device to use for calculations
    """
    faithfulness_loss = torch.tensor(0.0, device=device)
    scaled_faithfulness_loss = torch.tensor(0.0, device=device)
    for name in params1:
        diff = params2[name] - params1[name]
        faithfulness_loss += (diff ** 2).sum()
        scaled_faithfulness_loss += (scalers[name] * (diff ** 2)).sum()
    return faithfulness_loss / n_params, scaled_faithfulness_loss / n_params


def calc_faithfulness_loss(
    components: dict[str, LinearComponent | EmbeddingComponent],
    target_model: nn.Module,
    scale_fn: Callable[[Float[Tensor, ""]], Float[Tensor, ""]],
    n_params: int,
    device: str,
) -> Tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    """Calculate the MSE loss between component parameters (A@B + bias) and target parameters."""
    target_params: dict[str, Float[Tensor, "d_in d_out"]] = {}
    component_params: dict[str, Float[Tensor, "d_in d_out"]] = {}
    scalers_params: dict[str, Float[Tensor, ""]] = {}

    for comp_name, component in components.items():
        submodule = target_model.get_submodule(comp_name)
        assert isinstance(submodule, nn.Linear | nn.Embedding)
        target_params[comp_name] = submodule.weight
        component_params[comp_name] = component.weight
        if component.filler_comp_bool:
            component_params[comp_name] += component.filler_comp_weight.T
        scalers_params[comp_name] = scale_fn(
            submodule.weight
        )
        assert component_params[comp_name].shape == target_params[comp_name].shape

    faithfulness_loss, scaled_faithfulness_loss = _calc_tensors_mse(
        params1=component_params,
        params2=target_params,
        scalers=scalers_params,
        n_params=n_params,
        device=device,
    )
    return faithfulness_loss, scaled_faithfulness_loss
def calc_ce_loss(pred: Float[Tensor, "... vocab"], batch: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    flat_batch = batch[:, 1:].flatten()
    flat_pred_logits = einops.rearrange(pred[:, :-1], "... vocab -> (...) vocab")
    return F.cross_entropy(input=flat_pred_logits, target=flat_batch)

def calc_ce_losses(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: dict[str, Float[Tensor, "..."]],
    all_components_logits: Float[Tensor, "..."],
    masked_component_logits: Float[Tensor, "..."],
    clamped_masked_component_logits: Float[Tensor, "..."],
    stochastic_component_logits: Float[Tensor, "..."],
    binned_masked_component_logits: Float[Tensor, "..."],
    noisy_masked_component_logits: Float[Tensor, "..."],
    target_logits: Float[Tensor, "..."],
    task: Literal["lm", "cv"] = "lm",
    labels: Int[Tensor, "..."] | None = None,
) -> dict[str, float]:
    """Calculate cross-entropy losses for various masking scenarios.

    Args:
        model: The component model
        batch: Input batch
        components: Dictionary of components
        masks: Dictionary of masks for components
        all_components_logits: Logits from all components
        masked_component_logits: Logits from masked components
        target_logits: Target model logits

    Returns:
        Dictionary containing CE losses for different scenarios
    """
    ce_losses = {}

    # Flatten logits and batch for CE calculation
    # CE when every component is fully masked (all-zero masks)
    zero_masks = {k: torch.zeros_like(v) for k, v in masks.items()}
    zero_masked_component_logits = model.forward_with_components(
        batch, components=components, masks=zero_masks
    )
    if task == "lm":
        # Remove the first token from the batch (since it's not predicted)
        flat_batch = batch[:, 1:].flatten()

        # Remove the last prediction (since there's no ground truth token for it)
        flat_zero_masked_component_logits = einops.rearrange(zero_masked_component_logits[:, :-1], "... vocab -> (...) vocab")
        flat_all_component_logits = einops.rearrange( all_components_logits[:, :-1], "... vocab -> (...) vocab")
        flat_masked_component_logits = einops.rearrange(masked_component_logits[:, :-1], "... vocab -> (...) vocab")
        flat_stochastic_component_logits = einops.rearrange(stochastic_component_logits[:, :-1], "... vocab -> (...) vocab")
        flat_clamped_masked_component_logits = einops.rearrange(clamped_masked_component_logits[:, :-1], "... vocab -> (...) vocab")
        flat_binned_masked_component_logits = einops.rearrange(binned_masked_component_logits[:, :-1], "... vocab -> (...) vocab")
        flat_noisy_masked_component_logits = einops.rearrange(noisy_masked_component_logits[:, :-1], "... vocab -> (...) vocab")
        flat_target_logits = einops.rearrange(target_logits[:, :-1], "... vocab -> (...) vocab")

        all_components_ce_loss = F.cross_entropy(input=flat_all_component_logits, target=flat_batch)
        masked_ce_loss = F.cross_entropy(input=flat_masked_component_logits, target=flat_batch)
        stochastic_ce_loss = F.cross_entropy(input=flat_stochastic_component_logits, target=flat_batch)
        clamped_ce_loss = F.cross_entropy(input=flat_clamped_masked_component_logits, target=flat_batch)
        binned_ce_loss = F.cross_entropy(input=flat_binned_masked_component_logits, target=flat_batch)
        zero_masked_ce_loss = F.cross_entropy(input=flat_zero_masked_component_logits, target=flat_batch)
        noisy_ce_loss = F.cross_entropy(input=flat_noisy_masked_component_logits, target=flat_batch)
        target_ce_loss = F.cross_entropy(input=flat_target_logits, target=flat_batch)

    elif task == "cv":
        assert labels is not None, "Labels must be provided for classification tasks"
        # For classification tasks, we assume the logits are already in the correct format
        # and we can directly calculate CE losses.
        all_components_ce_loss = F.cross_entropy(input=all_components_logits, target=labels)
        masked_ce_loss = F.cross_entropy(input=masked_component_logits, target=labels)
        stochastic_ce_loss = F.cross_entropy(input=stochastic_component_logits, target=labels)
        clamped_ce_loss = F.cross_entropy(input=clamped_masked_component_logits, target=labels)
        binned_ce_loss = F.cross_entropy(input=binned_masked_component_logits, target=labels)
        zero_masked_ce_loss = F.cross_entropy(input=zero_masked_component_logits, target=labels)
        noisy_ce_loss = F.cross_entropy(input=noisy_masked_component_logits, target=labels)
        target_ce_loss = F.cross_entropy(input=target_logits, target=labels)

    ce_losses["ce/all_components_ce_diff"] = all_components_ce_loss.item() - target_ce_loss.item()
    ce_losses["ce/masked_ce_diff"] = masked_ce_loss.item() - target_ce_loss.item()
    ce_losses["ce/stochastic_ce_diff"] = stochastic_ce_loss.item() - target_ce_loss.item()
    ce_losses["ce/clamped_ce_diff"] = clamped_ce_loss.item() - target_ce_loss.item()
    ce_losses["ce/binned_ce_diff"] = binned_ce_loss.item() - target_ce_loss.item()
    ce_losses["ce/zero_masked_ce_diff"] = zero_masked_ce_loss.item() - target_ce_loss.item()
    ce_losses["ce/noisy_ce_diff"] = noisy_ce_loss.item() - target_ce_loss.item()

    return ce_losses

def calc_accuracies(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: dict[str, Float[Tensor, "..."]],
    all_components_logits: Float[Tensor, "..."],
    masked_component_logits: Float[Tensor, "..."],
    clamped_masked_component_logits: Float[Tensor, "..."],
    stochastic_component_logits: Float[Tensor, "..."],
    binned_masked_component_logits: Float[Tensor, "..."],
    noisy_masked_component_logits: Float[Tensor, "..."],
    target_logits: Float[Tensor, "..."],
    task: Literal["lm", "cv"] = "lm",
    labels: Int[Tensor, "..."] | None = None,
) -> dict[str, float]:
    """Calculate accuracies for various masking scenarios.

    Args:
        model: The component model
        batch: Input batch
        components: Dictionary of components
        masks: Dictionary of masks for components
        all_components_logits: Logits from all components
        masked_component_logits: Logits from masked components
        target_logits: Target model logits

    Returns:
        Dictionary containing accuracies for different scenarios
    """
    accuracies = {}

    # Accuracy when every component is fully masked (all-zero masks)
    zero_masks = {k: torch.zeros_like(v) for k, v in masks.items()}
    zero_masked_component_logits = model.forward_with_components(
        batch, components=components, masks=zero_masks
    )

    if task == "lm":
        # Flatten logits and batch for accuracy calculation
        flat_all_component_logits = einops.rearrange(
            all_components_logits[:, :-1], "... vocab -> (...) vocab"
        )
        flat_masked_component_logits = einops.rearrange(
            masked_component_logits[:, :-1], "... vocab -> (...) vocab"
        )
        flat_clamped_masked_component_logits = einops.rearrange(
            clamped_masked_component_logits[:, :-1], "... vocab -> (...) vocab"
        )
        flat_stochastic_component_logits = einops.rearrange(
            stochastic_component_logits[:, :-1], "... vocab -> (...) vocab"
        )
        flat_zero_masked_component_logits = einops.rearrange(
            zero_masked_component_logits[:, :-1], "... vocab -> (...) vocab"
        )
        flat_binned_masked_component_logits = einops.rearrange(
            binned_masked_component_logits[:, :-1], "... vocab -> (...) vocab"
        )
        flat_noisy_masked_component_logits = einops.rearrange(
            noisy_masked_component_logits[:, :-1], "... vocab -> (...) vocab"
        )
        flat_batch = batch[:, 1:].flatten()

        # Accuracy vs true labels
        all_components_accuracy = (flat_all_component_logits[:-1].argmax(dim=-1) == flat_batch).float().mean()
        masked_accuracy = (flat_masked_component_logits[:-1].argmax(dim=-1) == flat_batch).float().mean()
        target_accuracy = (target_logits[:-1].argmax(dim=-1) == flat_batch).float().mean()
        clamped_accuracy = (flat_clamped_masked_component_logits[:-1].argmax(dim=-1) == flat_batch).float().mean()
        stochastic_accuracy = (flat_stochastic_component_logits[:-1].argmax(dim=-1) == flat_batch).float().mean()
        zero_masked_accuracy = (flat_zero_masked_component_logits[:-1].argmax(dim=-1) == flat_batch).float().mean()
        binned_accuracy = (flat_binned_masked_component_logits[:-1].argmax(dim=-1) == flat_batch).float().mean()
        noisy_accuracy = (flat_noisy_masked_component_logits[:-1].argmax(dim=-1) == flat_batch).float().mean()

        accuracies["acc/all_components_accuracy_vs_labels"] = all_components_accuracy.item()
        accuracies["acc/masked_accuracy_vs_labels"] = masked_accuracy.item()
        accuracies["acc/target_accuracy_vs_labels"] = target_accuracy.item()
        accuracies["acc/clamped_accuracy_vs_labels"] = clamped_accuracy.item()
        accuracies["acc/stochastic_accuracy_vs_labels"] = stochastic_accuracy.item()
        accuracies["acc/zero_masked_accuracy_vs_labels"] = zero_masked_accuracy.item()
        accuracies["acc/binned_accuracy_vs_labels"] = binned_accuracy.item()
        accuracies["acc/noisy_accuracy_vs_labels"] = noisy_accuracy.item()
    elif task == "cv":
        # For classification tasks, we assume the logits are already in the correct format
        # and we can directly calculate accuracies.
        all_components_accuracy = (all_components_logits.argmax(dim=-1) == labels).float().mean()
        masked_accuracy = (masked_component_logits.argmax(dim=-1) == labels).float().mean()
        target_accuracy = (target_logits.argmax(dim=-1) == labels).float().mean()
        zero_masked_accuracy = (zero_masked_component_logits.argmax(dim=-1) == labels).float().mean()
        binned_accuracy = (binned_masked_component_logits.argmax(dim=-1) == labels).float().mean()
        clamped_accuracy = (clamped_masked_component_logits.argmax(dim=-1) == labels).float().mean()
        stochastic_accuracy = (stochastic_component_logits.argmax(dim=-1) == labels).float().mean()
        noisy_accuracy = (noisy_masked_component_logits.argmax(dim=-1) == labels).float().mean()

        accuracies["acc/all_components_accuracy_vs_labels"] = all_components_accuracy.item()
        accuracies["acc/masked_accuracy_vs_labels"] = masked_accuracy.item()
        accuracies["acc/target_accuracy_vs_labels"] = target_accuracy.item()
        accuracies["acc/zero_masked_accuracy_vs_labels"] = zero_masked_accuracy.item()
        accuracies["acc/binned_accuracy_vs_labels"] = binned_accuracy.item()
        accuracies["acc/clamped_accuracy_vs_labels"] = clamped_accuracy.item()
        accuracies["acc/stochastic_accuracy_vs_labels"] = stochastic_accuracy.item()
        accuracies["acc/noisy_accuracy_vs_labels"] = noisy_accuracy.item()
        
    return accuracies


def calculate_losses(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    config: Config,
    components: dict[str, LinearComponent | EmbeddingComponent],
    causal_importances: dict[str, Float[Tensor, "batch C"]],
    causal_importances_upper_leaky: dict[str, Float[Tensor, "batch C"]],
    target_out: Tensor,
    device: str,
    n_params: int,
    faithfulness_scale_fn: Callable[..., Float[Tensor, ""]],
    pnorm: float,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
    """Calculate all losses and return total loss and individual loss terms.

    Args:
        model: The component model
        batch: Input batch
        config: Configuration object with loss coefficients
        components: Dictionary of component modules
        causal_importances: Causal importance masks
        causal_importances_upper_leaky: Upper leaky causal importances for regularization
        binned_causal_importances: Binned causal importances for regularization
        target_out: Target model output
        device: Device to run computations on
        n_params: Total number of parameters in the model
        faithfulness_scale_fn: Function to scale the faithfulness loss

    Returns:
        Tuple of (total_loss, loss_terms_dict)
    """
    total_loss = torch.tensor(0.0, device=device)
    loss_terms = {}

    # Faithfulness loss
    faithfulness_loss, scaled_faithfulness_loss = calc_faithfulness_loss(
        components=components, target_model=model.model, n_params=n_params, scale_fn=faithfulness_scale_fn, device=device
    )
    total_loss += config.faithfulness_coeff * scaled_faithfulness_loss
    loss_terms["loss/faithfulness"] = faithfulness_loss.item()
    loss_terms["loss/scaled_faithfulness"] = scaled_faithfulness_loss.item()
    
    if "ce_labels" in config.output_loss_types:
        # Remove the first token from the batch (since it's not predicted)
        flat_batch = batch[:, 1:].flatten()
        flat_target_logits = einops.rearrange(target_out[:, :-1], "... vocab -> (...) vocab")
        target_ce_loss = F.cross_entropy(input=flat_target_logits, target=flat_batch)
    else:
        target_ce_loss = None

    # Reconstruction loss
    if config.recon_coeff is not None:
        recon_loss = calc_masked_recon_loss(
            model=model,
            batch=batch,
            components=components,
            masks=causal_importances,
            target_out=target_out,
            loss_type=config.output_loss_type,
        )
        total_loss += config.recon_coeff * recon_loss
        loss_terms["loss/recon"] = recon_loss.item()

    if config.clamped_recon_coeff is not None:
        clamped_masks = {k: torch.clamp(v, min=0, max=1) for k, v in causal_importances.items()}
        # Forward pass with components
        clamped_logits = model.forward_with_components(batch, components=components, masks=clamped_masks, filler_comp_scalar=0)
        
        # Calculate all requested losses
        losses = calculate_possible_losses(
            pred=clamped_logits,
            target=target_out,
            config=config,
            batch=batch,
            target_ce_loss=target_ce_loss,
        )
        for loss_type in config.output_loss_types:
            loss_terms[f"loss/clamped_recon_{loss_type}"] = losses[loss_type].item()
            total_loss += config.clamped_recon_coeff * abs(losses[loss_type]) * config.output_loss_types[loss_type].weight

        if config.filler_recon_coeff is not None and config.learned_filler_comp:
            clamped_logits_with_filler = model.forward_with_components(batch, components=components, masks=clamped_masks, filler_comp_scalar=1)
            clamped_recon_loss_with_filler = calc_kl_divergence_lm(pred=clamped_logits_with_filler, target=clamped_logits)
            total_loss += config.filler_recon_coeff * clamped_recon_loss_with_filler
            loss_terms["loss/clamped_recon_filler_kl"] = clamped_recon_loss_with_filler.item()

    # # Clamped reconstruction loss
    # if config.clamped_recon_coeff is not None:
    #     clamped_masks = {k: torch.clamp(v, min=0, max=1) for k, v in causal_importances.items()}
    #     if config.filler_recon_coeff is not None and config.learned_filler_comp:
    #         # Then we need to do Recon no Filler as target, & Recon with Filler as KL w/ no-filler
    #         clamped_logits = model.forward_with_components(batch, components=components, masks=clamped_masks, filler_comp_scalar=0)
    #         clamped_logits_with_filler = model.forward_with_components(batch, components=components, masks=clamped_masks, filler_comp_scalar=1)
    #         clamped_recon_loss_with_filler = calc_kl_divergence_lm(pred=clamped_logits_with_filler, target=clamped_logits)
    #         if config.output_loss_type == "kl":
    #             clamped_recon_loss = calc_kl_divergence_lm(pred=clamped_logits, target=target_out)
    #             total_loss += config.clamped_recon_coeff * clamped_recon_loss
    #             loss_terms["loss/clamped_recon_kl"] = clamped_recon_loss.item()
    #         elif config.output_loss_type == "ce-diff":
    #             flat_clamped_masked_component_logits = einops.rearrange(clamped_logits[:, :-1], "... vocab -> (...) vocab")
    #             clamped_ce_loss = F.cross_entropy(input=flat_clamped_masked_component_logits, target=flat_batch)
    #             total_loss += config.clamped_recon_coeff * clamped_ce_loss
    #             loss_terms["loss/clamped_recon_ce_diff"] = clamped_ce_loss.item() - target_ce_loss.item()
    #         # Filler loss is KL
    #         total_loss += config.filler_recon_coeff * clamped_recon_loss_with_filler
    #         loss_terms["loss/clamped_recon_filler_kl"] = clamped_recon_loss_with_filler.item()
    #     else:
    #         clamped_recon_loss = calc_masked_recon_loss(
    #             model=model,
    #             batch=batch,
    #             components=components,
    #             masks=clamped_masks,
    #             target_out=target_out,  
    #             loss_type=config.output_loss_type,
    #         )
    #         total_loss += config.clamped_recon_coeff * clamped_recon_loss
    #         loss_terms["loss/clamped_recon"] = clamped_recon_loss.item()


    if config.stochastic_recon_coeff is not None:
        stochastic_masks = calc_stochastic_masks(
            causal_importances=causal_importances, n_mask_samples=config.n_mask_samples
        )
        stochastic_recon_loss = torch.tensor(0.0, device=target_out.device)
        for i in range(len(stochastic_masks)):
            stochastic_masks[i] = {k: torch.clamp(v, min=0, max=1) for k, v in stochastic_masks[i].items()}
            stochastic_logits = model.forward_with_components(batch, components=components, masks=stochastic_masks[i], filler_comp_scalar=0.0,
            )
            losses = calculate_possible_losses(
                pred=stochastic_logits,
                target=target_out,
                config=config,
                batch=batch,
                target_ce_loss=target_ce_loss,
            )
            for loss_type in config.output_loss_types:
                loss_terms[f"loss/stochastic_recon_{loss_type}"] = losses[loss_type].item()
                total_loss += config.stochastic_recon_coeff * abs(losses[loss_type]) * config.output_loss_types[loss_type].weight

    
    # # Stochastic reconstruction loss
    # if config.stochastic_recon_coeff is not None:
    #     stochastic_masks = calc_stochastic_masks(
    #         causal_importances=causal_importances, n_mask_samples=config.n_mask_samples
    #     )
    #     stochastic_recon_loss = torch.tensor(0.0, device=target_out.device)
    #     for i in range(len(stochastic_masks)):
    #         if config.learned_filler_comp:
    #             stochastic_masks[i] = {k: torch.clamp(v, min=0, max=1) for k, v in stochastic_masks[i].items()}
    #             masked_logits_with_filler = model.forward_with_components(batch, components=components, masks=stochastic_masks[i], filler_comp_scalar=1)
    #             if config.output_loss_type == "kl":
    #                 stochastic_recon_loss += calc_kl_divergence_lm(pred=masked_logits_with_filler, target=target_out).item()
    #             elif config.output_loss_type == "ce-diff":
    #                 flat_stochastic_component_logits = einops.rearrange(flat_stochastic_component_logits[:, :-1], "... vocab -> (...) vocab")
    #                 stochastic_ce_loss = F.cross_entropy(input=flat_stochastic_component_logits, target=flat_batch)
    #                 stochastic_recon_loss += stochastic_ce_loss.item() - target_ce_loss.item()

    #         else:  
    #             stochastic_recon_loss += calc_masked_recon_loss(
    #                 model=model,
    #                 batch=batch,
    #                 components=components,
    #                 masks=stochastic_masks[i],
    #                 target_out=target_out,
    #                 loss_type=config.output_loss_type,
    #             )
    #     stochastic_recon_loss = stochastic_recon_loss / len(stochastic_masks)
    #     total_loss += config.stochastic_recon_coeff * stochastic_recon_loss
    #     if config.output_loss_type == "ce-diff":
    #         loss_terms["loss/stochastic_recon_ce_diff"] = stochastic_recon_loss.item()
    #     elif config.output_loss_type == "kl":
    #         loss_terms["loss/stochastic_recon_kl"] = stochastic_recon_loss.item()


    # Reconstruction layerwise loss
    if config.recon_layerwise_coeff is not None:
        recon_layerwise_loss = calc_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            device=device,
            components=components,
            masks=[causal_importances],
            target_out=target_out,
            loss_type=config.output_loss_type,
        )
        total_loss += config.recon_layerwise_coeff * recon_layerwise_loss
        loss_terms["loss/recon_layerwise"] = recon_layerwise_loss.item()

    # Stochastic reconstruction layerwise loss
    if config.stochastic_recon_layerwise_coeff is not None:
        layerwise_stochastic_masks = calc_stochastic_masks(
            causal_importances=causal_importances, n_mask_samples=config.n_mask_samples
        )
        stochastic_recon_layerwise_loss = calc_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            device=device,
            components=components,
            masks=layerwise_stochastic_masks,
            target_out=target_out,
            loss_type=config.output_loss_type,
        )
        total_loss += config.stochastic_recon_layerwise_coeff * stochastic_recon_layerwise_loss
        loss_terms["loss/stochastic_recon_layerwise"] = stochastic_recon_layerwise_loss.item()

    # Importance minimality loss
    importance_minimality_loss = calc_importance_minimality_loss(
        ci_upper_leaky=causal_importances_upper_leaky, pnorm=pnorm
    )
    total_loss += config.importance_minimality_coeff * importance_minimality_loss
    loss_terms["loss/importance_minimality"] = importance_minimality_loss.item()

    # Schatten loss
    if config.schatten_coeff is not None:
        schatten_loss = calc_schatten_loss(
            ci_upper_leaky=causal_importances_upper_leaky,
            pnorm=pnorm,
            components=components,
            device=device,
        )
        total_loss += config.schatten_coeff * schatten_loss
        loss_terms["loss/schatten"] = schatten_loss.item()

    if config.all_components_recon_coeff is not None:
        masks_all_ones = {k: torch.ones_like(v) for k, v in causal_importances.items()}
        all_ones_logits = model.forward_with_components(batch, components=components, masks=masks_all_ones, filler_comp_scalar=1 if config.learned_filler_comp else 0.0)
        losses = calculate_possible_losses(
            pred=all_ones_logits,
            target=target_out,
            config=config,
            batch=batch,
            target_ce_loss=target_ce_loss,
        )
        for loss_type in config.output_loss_types:
            loss_terms[f"loss/all_components_recon_{loss_type}"] = losses[loss_type].item()
            total_loss += config.all_components_recon_coeff * abs(losses[loss_type]) * config.output_loss_types[loss_type].weight

    # # Output reconstruction loss
    # if config.all_components_recon_coeff is not None:
    #     masks_all_ones = {k: torch.ones_like(v) for k, v in causal_importances.items()}
    #     if config.learned_filler_comp:
    #         all_ones_logits = model.forward_with_components(batch, components=components, masks=masks_all_ones, filler_comp_scalar=1)
    #         if config.output_loss_type == "kl":
    #             out_recon_loss = calc_kl_divergence_lm(pred=all_ones_logits, target=target_out)
    #             loss_terms["loss/all_components_recon_kl"] = out_recon_loss.item()
    #         elif config.output_loss_type == "ce-diff":
    #             # Remove the last prediction (since there's no ground truth token for it)
    #             flat_all_component_logits = einops.rearrange(all_ones_logits[:, :-1], "... vocab -> (...) vocab")
    #             all_components_ce_loss = F.cross_entropy(input=flat_all_component_logits, target=flat_batch)
    #             out_recon_loss = all_components_ce_loss.item() - target_ce_loss.item()
    #             loss_terms["loss/all_components_recon_ce_diff"] = out_recon_loss.item()
    #         else:
    #             raise ValueError(f"Loss type {config.output_loss_type} not supported")
    #         total_loss += config.all_components_recon_coeff * out_recon_loss
    #         # loss_terms["loss/all_components_recon"] = out_recon_loss.item()
    #     else:
    #         out_recon_loss = calc_masked_recon_loss(
    #             model=model,
    #             batch=batch,
    #             components=components,
    #             masks=masks_all_ones,
    #             target_out=target_out,
    #             loss_type=config.output_loss_type,
    #         )
    #         total_loss += config.all_components_recon_coeff * out_recon_loss
    #         loss_terms["loss/all_components_recon"] = out_recon_loss.item()

    # Embedding reconstruction loss
    if config.embedding_recon_coeff is not None:
        stochastic_masks = calc_stochastic_masks(
            causal_importances=causal_importances, n_mask_samples=config.n_mask_samples
        )
        assert len(components) == 1, "Only one embedding component is supported"
        component = list(components.values())[0]
        assert isinstance(component, EmbeddingComponent)
        embedding_recon_loss = calc_embedding_recon_loss(
            model=model,
            batch=batch,
            component=component,
            masks=stochastic_masks,
            embed_module_name=next(iter(components.keys())),
            unembed=config.is_embed_unembed_recon,
        )
        total_loss += config.embedding_recon_coeff * embedding_recon_loss
        loss_terms["loss/embedding_recon"] = embedding_recon_loss.item()

    return total_loss, loss_terms
