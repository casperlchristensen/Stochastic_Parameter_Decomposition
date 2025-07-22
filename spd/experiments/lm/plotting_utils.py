"""
Plotting utilities for component model analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple


def plot_encoder_decoder_norms(
    enc_norms: torch.Tensor,
    dec_norms: torch.Tensor, 
    alive_mask: torch.Tensor,
    dead_mask: torch.Tensor,
    same_scale: bool = False,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot encoder vs decoder norms, colored by alive/dead status.
    
    Args:
        enc_norms: Encoder L2 norms
        dec_norms: Decoder L2 norms
        alive_mask: Boolean mask for alive components
        dead_mask: Boolean mask for dead components
        same_scale: Whether to use same scale for x and y axes
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get alive and dead norms
    alive_enc_norms = enc_norms[alive_mask].cpu().numpy()
    alive_dec_norms = dec_norms[alive_mask].cpu().numpy()
    dead_enc_norms = enc_norms[dead_mask].cpu().numpy()
    dead_dec_norms = dec_norms[dead_mask].cpu().numpy()
    
    # Plot
    ax.scatter(dead_enc_norms, dead_dec_norms, alpha=0.5, 
               label="Dead Components", color="red")
    ax.scatter(alive_enc_norms, alive_dec_norms, alpha=0.5, 
               label="Alive Components", color="blue")
    
    ax.set_xlabel("Encoder Norm")
    ax.set_ylabel("Decoder Norm")
    ax.set_title("Encoder vs Decoder Norms")
    
    # Set axis limits
    if same_scale:
        max_norm = max(enc_norms.max().item(), dec_norms.max().item())
        ax.set_xlim(0, max_norm)
        ax.set_ylim(0, max_norm)
    else:
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_frequency_vs_weight_sim(
    frequency: torch.Tensor,
    weight_sim_importance: torch.Tensor,
    dec_norms: torch.Tensor,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot component frequency vs weight similarity importance,
    colored by decoder norm.
    
    Args:
        frequency: Component activation frequencies
        weight_sim_importance: Weight similarity importance scores
        dec_norms: Decoder norms for coloring
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        frequency.cpu().numpy(),
        weight_sim_importance.cpu().numpy(),
        c=dec_norms.cpu().numpy(),
        cmap='viridis',
        alpha=0.6
    )
    
    ax.set_xlabel("Component Frequency")
    ax.set_ylabel("Weight Sim Importance")
    ax.set_title("Weight Sim Importance vs Component Frequency")
    ax.set_xscale("log")
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Decoder Norm")
    cbar.mappable.set_clim(vmin=0)
    
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_weight_sim_histogram(
    weight_sim_importance: torch.Tensor,
    alive_mask: torch.Tensor,
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot histogram of weight similarity importance for alive vs dead components.
    
    Args:
        weight_sim_importance: Weight similarity importance scores
        alive_mask: Boolean mask for alive components
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    dead_mask = ~alive_mask
    weight_sim_imp_dead = weight_sim_importance[dead_mask].cpu().numpy()
    weight_sim_imp_alive = weight_sim_importance[alive_mask].cpu().numpy()
    
    ax.hist(weight_sim_imp_dead, bins=bins, alpha=0.5, 
            label="Dead Components", color="red")
    ax.hist(weight_sim_imp_alive, bins=bins, alpha=0.5, 
            label="Alive Components", color="blue")
    
    ax.set_xlabel("Weight Sim Importance")
    ax.set_ylabel("Count")
    ax.set_title("Weight Sim Importance Distribution")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_mmcs_histograms(
    mmcs_metrics: Dict[str, Dict[str, torch.Tensor]],
    plot_type: str = 'max',
    bins: int = 25,
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Plot histograms of MMCS values for encoders and decoders.
    
    Args:
        mmcs_metrics: Dictionary of MMCS metrics from calculate_all_mmcs_metrics
        plot_type: 'max' or 'min'
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    metrics = mmcs_metrics[plot_type]
    
    # Encoder plot
    ax1.hist(metrics['mmcs_a_a_enc'].cpu().numpy(), bins=bins, alpha=0.5, 
             label="Alive-Alive Enc")
    ax1.hist(metrics['mmcs_d_d_enc'].cpu().numpy(), bins=bins, alpha=0.5, 
             label="Dead-Dead Enc")
    ax1.hist(metrics['mmcs_d_a_enc'].cpu().numpy(), bins=bins, alpha=0.5, 
             label="Dead-Alive Enc")
    ax1.set_title(f"{plot_type.capitalize()} Cosine Similarity - Encoders")
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_ylabel("Count")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if plot_type == 'max':
        ax1.set_xlim(0, 1)
    else:
        ax1.set_xlim(-1, 0)
    
    # Decoder plot
    ax2.hist(metrics['mmcs_a_a_dec'].cpu().numpy(), bins=bins, alpha=0.5, 
             label="Alive-Alive Dec")
    ax2.hist(metrics['mmcs_d_d_dec'].cpu().numpy(), bins=bins, alpha=0.5, 
             label="Dead-Dead Dec")
    ax2.hist(metrics['mmcs_d_a_dec'].cpu().numpy(), bins=bins, alpha=0.5, 
             label="Dead-Alive Dec")
    ax2.set_title(f"{plot_type.capitalize()} Cosine Similarity - Decoders")
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if plot_type == 'max':
        ax2.set_xlim(0, 1)
    else:
        ax2.set_xlim(-1, 0)
    
    plt.tight_layout()
    return fig


def plot_alive_frequency_relationships(
    alive_freq: torch.Tensor,
    alive_enc_norms: torch.Tensor,
    alive_dec_norms: torch.Tensor,
    mmcs_a_a_enc: Optional[torch.Tensor] = None,
    mmcs_a_a_dec: Optional[torch.Tensor] = None,
    weight_sim_imp_alive: Optional[torch.Tensor] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot relationships between frequency and various metrics for alive components.
    
    Args:
        alive_freq: Frequencies of alive components
        alive_enc_norms: Encoder norms of alive components
        alive_dec_norms: Decoder norms of alive components
        mmcs_a_a_enc: Max cosine similarity for alive encoders (optional)
        mmcs_a_a_dec: Max cosine similarity for alive decoders (optional)
        weight_sim_imp_alive: Weight sim importance for alive components (optional)
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    n_plots = 2 + (mmcs_a_a_enc is not None) + (mmcs_a_a_dec is not None)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    plot_idx = 0
    
    # Frequency vs decoder norm, colored by encoder norm
    scatter1 = axes[plot_idx].scatter(
        alive_freq.cpu().numpy(),
        alive_dec_norms.cpu().numpy(),
        c=alive_enc_norms.cpu().numpy(),
        cmap='viridis',
        alpha=0.6
    )
    axes[plot_idx].set_xlabel("Frequency")
    axes[plot_idx].set_ylabel("Decoder Norm")
    axes[plot_idx].set_xscale("log")
    axes[plot_idx].set_title("Alive Components: Frequency vs Decoder Norm")
    cbar1 = plt.colorbar(scatter1, ax=axes[plot_idx])
    cbar1.set_label("Encoder Norm")
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1
    
    # Encoder vs decoder norms, colored by frequency
    scatter2 = axes[plot_idx].scatter(
        alive_enc_norms.cpu().numpy(),
        alive_dec_norms.cpu().numpy(),
        c=alive_freq.cpu().numpy(),
        cmap='plasma',
        alpha=0.6
    )
    axes[plot_idx].set_xlabel("Encoder Norm")
    axes[plot_idx].set_ylabel("Decoder Norm")
    axes[plot_idx].set_title("Alive Components: Encoder vs Decoder Norm")
    cbar2 = plt.colorbar(scatter2, ax=axes[plot_idx])
    cbar2.set_label("Frequency (log scale)")
    cbar2.mappable.set_clim(vmin=alive_freq.min().item(), vmax=alive_freq.max().item())
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1
    
    # MMCS plots if provided
    if mmcs_a_a_dec is not None and weight_sim_imp_alive is not None:
        scatter3 = axes[plot_idx].scatter(
            alive_freq.cpu().numpy(),
            mmcs_a_a_dec.cpu().numpy(),
            c=weight_sim_imp_alive.cpu().numpy(),
            cmap='viridis',
            alpha=0.6
        )
        axes[plot_idx].set_xlabel("Frequency")
        axes[plot_idx].set_ylabel("Max Cosine Sim (Dec)")
        axes[plot_idx].set_xscale("log")
        axes[plot_idx].set_title("Alive-Alive Dec vs Frequency")
        cbar3 = plt.colorbar(scatter3, ax=axes[plot_idx])
        cbar3.set_label("Weight Sim Importance")
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    if mmcs_a_a_enc is not None and weight_sim_imp_alive is not None:
        scatter4 = axes[plot_idx].scatter(
            alive_freq.cpu().numpy(),
            mmcs_a_a_enc.cpu().numpy(),
            c=weight_sim_imp_alive.cpu().numpy(),
            cmap='viridis',
            alpha=0.6
        )
        axes[plot_idx].set_xlabel("Frequency")
        axes[plot_idx].set_ylabel("Max Cosine Sim (Enc)")
        axes[plot_idx].set_xscale("log")
        axes[plot_idx].set_title("Alive-Alive Enc vs Frequency")
        cbar4 = plt.colorbar(scatter4, ax=axes[plot_idx])
        cbar4.set_label("Weight Sim Importance")
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Hide unused axes
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_correlation_summary(
    correlations: Dict[str, float],
    title: str = "Correlation Summary",
    figsize: Tuple[int, int] = (10, 6),
    highlight_threshold: float = 0.5
) -> plt.Figure:
    """
    Plot a bar chart of correlations with highlighting for strong correlations.
    
    Args:
        correlations: Dictionary of correlation values
        title: Plot title
        figsize: Figure size
        highlight_threshold: Threshold for highlighting strong correlations
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by absolute correlation value
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    labels = [item[0] for item in sorted_corrs]
    values = [item[1] for item in sorted_corrs]
    
    # Color based on correlation strength
    colors = ['red' if abs(v) > highlight_threshold else 'blue' for v in values]
    
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom' if height > 0 else 'top')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Correlation")
    ax.set_title(title)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=highlight_threshold, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-highlight_threshold, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig