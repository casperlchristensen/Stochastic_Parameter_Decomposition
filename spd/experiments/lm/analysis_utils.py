"""
Utility functions for analyzing component models, including frequency analysis,
weight similarity, and cosine similarity metrics.
"""

import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import einops


@dataclass
class ComponentAnalysisResults:
    """Container for all component analysis results."""
    # Basic statistics
    frequency: Optional[torch.Tensor] = None
    weight_sim_importance: Optional[torch.Tensor] = None
    alive_mask: Optional[torch.Tensor] = None
    dead_mask: Optional[torch.Tensor] = None
    enc_norms: Optional[torch.Tensor] = None
    dec_norms: Optional[torch.Tensor] = None
    
    # Cosine similarities
    cosine_similarities: Optional[Dict] = None
    mmcs_metrics: Optional[Dict] = None
    
    # To be added later
    kl_statistics: Optional[Dict] = None
    dead_component_importance: Optional[Dict] = None
    
    # Module information
    module_name: Optional[str] = None
    num_components: Optional[int] = None


def calculate_component_frequency(
    model, 
    eval_loader, 
    components, 
    As, 
    gates, 
    module_comp, 
    module_name,
    threshold=0.0,
    device='cuda'
) -> torch.Tensor:
    """
    Calculate how frequently each component activates across the dataset.
    
    Args:
        model: The component model
        eval_loader: DataLoader for evaluation data
        components: Dictionary of component modules
        As: Dictionary of A matrices for components
        gates: Dictionary of gate modules
        module_comp: List of module names for forward hooks
        module_name: Specific module to analyze
        device: Device to run on
        
    Returns:
        frequency: Tensor of shape [num_components] with activation frequencies
    """
    from spd.models.component_utils import calc_causal_importances
    
    num_components = components[module_name].A.shape[-1]
    component_freq = torch.zeros(num_components)
    total_data_points = 0
    
    dataset = iter(eval_loader)
    
    for batch in tqdm(dataset, desc="Calculating component frequencies"):
        batch = batch["input_ids"].to(device)
        
        # Get pre-activation values
        orig_logits, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch,
            module_names=module_comp
        )
        
        # Calculate causal importances
        causal_importances, causal_importances_upper_leaky = calc_causal_importances(
            pre_weight_acts=pre_weight_acts, 
            As=As, 
            gates=gates, 
            detach_inputs=False
        )
        
        # Count activations
        activated = (causal_importances[module_name] > threshold).sum(dim=(0, 1)).cpu()
        component_freq += activated
        total_data_points += batch.numel()
    
    # Convert to frequency
    frequency = component_freq / total_data_points
    
    return frequency


def calculate_weight_sim_importance(
    components, 
    model, 
    num_components, 
    device='cuda'
) -> torch.Tensor:
    """
    Calculate weight similarity importance for each component.
    This measures how removing each component affects weight reconstruction.
    
    Args:
        components: Dictionary of component modules
        model: The base model
        num_components: Total number of components
        device: Device to run on
        
    Returns:
        weight_sim_importances: Tensor of importance scores for each component
    """
    def weight_sim(components, model, exclude_mask=None):
        """Calculate weight similarity with optional component exclusion."""
        component_params = {}
        target_params = {}
        
        # Create exclusion mask if needed
        if exclude_mask is not None:
            if isinstance(exclude_mask, int):
                include_mask = torch.ones(num_components, dtype=torch.bool)
                include_mask[exclude_mask] = False
            else:
                include_mask = ~exclude_mask
        
        # Process each component
        for comp_name, component in components.items():
            if exclude_mask is not None:
                A_filtered = component.A[:, include_mask]
                B_filtered = component.B[include_mask]
                component_params[comp_name] = einops.einsum(
                    A_filtered, B_filtered,
                    "d_in C, C d_out -> d_out d_in"
                )
            else:
                component_params[comp_name] = einops.einsum(
                    component.A, component.B,
                    "d_in C, C d_out -> d_out d_in"
                )
            
            # Get target parameters from model
            submodule = model.model.get_submodule(comp_name)
            target_params[comp_name] = submodule.weight
        
        # Calculate faithfulness loss
        faithfulness_loss = torch.tensor(0.0, device=device)
        
        for name in component_params:
            diff = target_params[name] - component_params[name]
            faithfulness_loss += (diff ** 2).sum()
        
        return faithfulness_loss
    
    # Calculate importance scores
    weight_sim_importances = torch.zeros(num_components)
    original_loss = weight_sim(components, model)
    
    for i in tqdm(range(num_components), desc="Calculating weight sim importance"):
        weight_sim_importances[i] = weight_sim(components, model, i) - original_loss
    
    return weight_sim_importances


def calculate_encoder_decoder_norms(
    components, 
    module_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate L2 norms of encoder and decoder vectors.
    
    Args:
        components: Dictionary of component modules
        module_name: Name of the module to analyze
        
    Returns:
        enc_norms: Encoder norms [num_components]
        dec_norms: Decoder norms [num_components]
    """
    comp_encoders = components[module_name].A  # [d_model, n_components]
    comp_decoders = components[module_name].B  # [n_components, d_out]
    
    enc_norms = comp_encoders.norm(dim=0)
    dec_norms = comp_decoders.norm(dim=1)
    
    return enc_norms, dec_norms


def calculate_alive_dead_masks(
    frequency: torch.Tensor, 
    threshold: float = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create boolean masks for alive and dead components based on activation frequency.
    
    Args:
        frequency: Component activation frequencies
        threshold: Frequency threshold (components with freq > threshold are alive)
        
    Returns:
        alive_mask: Boolean mask for alive components
        dead_mask: Boolean mask for dead components
    """
    alive_mask = frequency > threshold
    dead_mask = frequency == 0
    
    return alive_mask, dead_mask


def calculate_cosine_similarities(
    components, 
    module_name: str, 
    alive_mask: torch.Tensor, 
    dead_mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Calculate cosine similarities between encoder and decoder vectors
    for alive and dead components.
    
    Args:
        components: Dictionary of component modules
        module_name: Name of the module to analyze
        alive_mask: Boolean mask for alive components
        dead_mask: Boolean mask for dead components
        
    Returns:
        Dictionary containing:
        - aa_cos_sim_alive: Alive-alive encoder cosine similarities
        - bb_cos_sim_alive: Alive-alive decoder cosine similarities
        - aa_cos_sim_dead: Dead-dead encoder cosine similarities
        - bb_cos_sim_dead: Dead-dead decoder cosine similarities
        - aa_cos_sim_alive_dead: Alive-dead encoder cosine similarities
        - bb_cos_sim_alive_dead: Alive-dead decoder cosine similarities
    """
    # Get component vectors
    comp_encoders = components[module_name].A  # [d_model, n_components]
    comp_decoders = components[module_name].B  # [n_components, d_out]
    
    # Split into alive and dead
    A_alive = comp_encoders[:, alive_mask]
    B_alive = comp_decoders[alive_mask, :]
    A_dead = comp_encoders[:, dead_mask]
    B_dead = comp_decoders[dead_mask, :]
    
    # Normalize vectors
    normed_A_alive = torch.nn.functional.normalize(A_alive, dim=0)
    normed_B_alive = torch.nn.functional.normalize(B_alive, dim=1)
    normed_A_dead = torch.nn.functional.normalize(A_dead, dim=0)
    normed_B_dead = torch.nn.functional.normalize(B_dead, dim=1)
    
    # Compute cosine similarities
    results = {}
    
    # Alive-alive similarities (lower triangular)
    results['aa_cos_sim_alive'] = (normed_A_alive.T @ normed_A_alive).tril(diagonal=-1)
    results['bb_cos_sim_alive'] = (normed_B_alive @ normed_B_alive.T).tril(diagonal=-1)
    
    # Dead-dead similarities (lower triangular)
    results['aa_cos_sim_dead'] = (normed_A_dead.T @ normed_A_dead).tril(diagonal=-1)
    results['bb_cos_sim_dead'] = (normed_B_dead @ normed_B_dead.T).tril(diagonal=-1)
    
    # Alive-dead cross similarities
    results['aa_cos_sim_alive_dead'] = normed_A_alive.T @ normed_A_dead
    results['bb_cos_sim_alive_dead'] = normed_B_alive @ normed_B_dead.T
    
    return results


def calculate_mmcs_metrics(
    vectors1: torch.Tensor, 
    vectors2: torch.Tensor, 
    compute_max: bool = True,
    lower_triangular: bool = True
) -> Tuple[torch.Tensor, float]:
    """
    Calculate Max/Min Mean Cosine Similarity (MMCS) between two sets of vectors.
    
    Args:
        vectors1: First set of vectors
        vectors2: Second set of vectors
        compute_max: If True, compute maximum; if False, compute minimum
        lower_triangular: If True, only consider lower triangular part
        
    Returns:
        per_component_values: Max/min cosine similarity for each component
        mean_value: Mean of the max/min values
    """
    # Normalize vectors
    if vectors1.shape[0] > vectors1.shape[1]:  # [d_model, n_components] format
        v1_norm = torch.nn.functional.normalize(vectors1, dim=0)
        v2_norm = torch.nn.functional.normalize(vectors2, dim=0)
        cos_sim = v1_norm.T @ v2_norm
    else:  # [n_components, d_out] format
        v1_norm = torch.nn.functional.normalize(vectors1, dim=1)
        v2_norm = torch.nn.functional.normalize(vectors2, dim=1)
        cos_sim = v1_norm.T @ v2_norm
    
    # Apply lower triangular mask if needed
    if lower_triangular:
        cos_sim = cos_sim.tril(diagonal=-1)
    
    # Compute max or min
    if compute_max:
        per_component_values = cos_sim.max(dim=0).values
    else:
        per_component_values = cos_sim.min(dim=0).values
    
    mean_value = per_component_values.mean().item()
    
    return per_component_values, mean_value


def calculate_all_mmcs_metrics(
    components,
    module_name: str,
    alive_mask: torch.Tensor,
    dead_mask: torch.Tensor
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Calculate all MMCS metrics for alive and dead components.
    
    Returns:
        Dictionary with 'max' and 'min' keys, each containing:
        - mmcs_a_a_enc/dec: Alive-alive MMCS values
        - mmcs_d_d_enc/dec: Dead-dead MMCS values  
        - mmcs_d_a_enc/dec: Dead-alive MMCS values
    """
    # Get component vectors
    comp_encoders = components[module_name].A
    comp_decoders = components[module_name].B
    
    # Split into alive and dead
    alive_enc = comp_encoders[:, alive_mask]
    alive_dec = comp_decoders[alive_mask, :]
    dead_enc = comp_encoders[:, dead_mask]
    dead_dec = comp_decoders[dead_mask, :]
    
    results = {'max': {}, 'min': {}}
    
    # Calculate all MMCS metrics
    for compute_max, key in [(True, 'max'), (False, 'min')]:
        # Alive-alive
        results[key]['mmcs_a_a_enc'], results[key]['mmcs_a_a_enc_mean'] = calculate_mmcs_metrics(
            alive_enc, alive_enc, compute_max=compute_max
        )
        results[key]['mmcs_a_a_dec'], results[key]['mmcs_a_a_dec_mean'] = calculate_mmcs_metrics(
            alive_dec.T, alive_dec.T, compute_max=compute_max
        )
        
        # Dead-dead
        results[key]['mmcs_d_d_enc'], results[key]['mmcs_d_d_enc_mean'] = calculate_mmcs_metrics(
            dead_enc, dead_enc, compute_max=compute_max
        )
        results[key]['mmcs_d_d_dec'], results[key]['mmcs_d_d_dec_mean'] = calculate_mmcs_metrics(
            dead_dec.T, dead_dec.T, compute_max=compute_max
        )
        
        # Dead-alive
        results[key]['mmcs_d_a_enc'], results[key]['mmcs_d_a_enc_mean'] = calculate_mmcs_metrics(
            alive_enc, dead_enc, compute_max=compute_max, lower_triangular=False
        )
        results[key]['mmcs_d_a_dec'], results[key]['mmcs_d_a_dec_mean'] = calculate_mmcs_metrics(
            alive_dec.T, dead_dec.T, compute_max=compute_max, lower_triangular=False
        )
    
    return results