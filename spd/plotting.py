import math
from collections.abc import Callable, Mapping
from typing import Literal

import matplotlib.ticker as tkr
import numpy as np
import torch
import wandb
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_causal_importances
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
    

def plot_all_cos_sims(components, ci, layer_name, alive_mask, config, log_data):
    frequencies = (ci >= 0.1).float().mean(dim=(0, 1))
    # Get alive and dead masks
    # alive_mask = components[layer_name].A.abs().sum(dim=0) > 0
    # alt_alive_mask = 
    dead_mask = ~alive_mask
    display_dead = dead_mask.sum() > 0 and not config.remove_dead_components
    
    
    # Split components into alive and dead
    A_alive = components[layer_name].A[:, alive_mask]
    B_alive = components[layer_name].B[alive_mask, :]
    A_dead = components[layer_name].A[:, dead_mask]
    B_dead = components[layer_name].B[dead_mask, :]
    
    alive_frequencies = frequencies[alive_mask].cpu().numpy()
    

    # Normalize alive components
    normed_A_alive = torch.nn.functional.normalize(A_alive, dim=0)  # [d_model, n_components_alive]
    normed_B_alive = torch.nn.functional.normalize(B_alive, dim=1)  # [n_components_alive, d_out]
    
    # Normalize dead components (if any exist)
    if display_dead:
        normed_A_dead = torch.nn.functional.normalize(A_dead, dim=0)  # [d_model, n_components_dead]
        normed_B_dead = torch.nn.functional.normalize(B_dead, dim=1)  # [n_components_dead, d_out]
    
    # Compute cosine similarities for alive components
    aa_cos_sim_alive = (normed_A_alive.T @ normed_A_alive).tril(diagonal=-1)
    bb_cos_sim_alive = (normed_B_alive @ normed_B_alive.T).tril(diagonal=-1)
    
    # Find max & min for alive components
    aa_max_alive = aa_cos_sim_alive.max(dim=0).values.cpu().numpy()
    aa_min_alive = aa_cos_sim_alive.min(dim=0).values.cpu().numpy()
    bb_max_alive = bb_cos_sim_alive.max(dim=0).values.cpu().numpy()
    bb_min_alive = bb_cos_sim_alive.min(dim=0).values.cpu().numpy()
    
    # If there are dead components, compute their similarities
    if display_dead:
        # Compute cosine similarities between alive and dead A components
        aa_cos_sim_alive_dead = normed_A_alive.T @ normed_A_dead  # [n_alive, n_dead]
        aa_max_alive_dead = aa_cos_sim_alive_dead.max(dim=1).values.cpu().numpy()  # max over dead for each alive
        aa_max_dead_alive = aa_cos_sim_alive_dead.max(dim=0).values.cpu().numpy()  # max over alive for each dead
        aa_min_alive_dead = aa_cos_sim_alive_dead.min(dim=1).values.cpu().numpy()  # min over dead for each alive
        aa_min_dead_alive = aa_cos_sim_alive_dead.min(dim=0).values.cpu().numpy()  # min over alive for each dead
        
        # Compute cosine similarities between alive and dead B components
        bb_cos_sim_alive_dead = normed_B_alive @ normed_B_dead.T  # [n_alive, n_dead]
        bb_max_alive_dead = bb_cos_sim_alive_dead.max(dim=1).values.cpu().numpy()  # max over dead for each alive
        bb_max_dead_alive = bb_cos_sim_alive_dead.max(dim=0).values.cpu().numpy()  # max over alive for each dead
        bb_min_alive_dead = bb_cos_sim_alive_dead.min(dim=1).values.cpu().numpy()  # min over dead for each alive
        bb_min_dead_alive = bb_cos_sim_alive_dead.min(dim=0).values.cpu().numpy()  # min over alive for each dead

        # Compute cosine similarities for dead components among themselves
        aa_cos_sim_dead = (normed_A_dead.T @ normed_A_dead).tril(diagonal=-1)
        bb_cos_sim_dead = (normed_B_dead @ normed_B_dead.T).tril(diagonal=-1)
        
        aa_max_dead = aa_cos_sim_dead.max(dim=0).values.cpu().numpy()
        aa_min_dead = aa_cos_sim_dead.min(dim=0).values.cpu().numpy()
        bb_max_dead = bb_cos_sim_dead.max(dim=0).values.cpu().numpy()
        bb_min_dead = bb_cos_sim_dead.min(dim=0).values.cpu().numpy()

    # Store in log_data
    log_data[f"cos_sim/{layer_name}_aa_max_alive_mean"] = aa_max_alive.mean()
    log_data[f"cos_sim/{layer_name}_aa_min_alive_mean"] = aa_min_alive.mean()
    log_data[f"cos_sim/{layer_name}_bb_max_alive_mean"] = bb_max_alive.mean()
    log_data[f"cos_sim/{layer_name}_bb_min_alive_mean"] = bb_min_alive.mean()


    # Build the correlation data - start with alive components
    correlation_dict = {
        'Frequency': alive_frequencies,
        'AA max (alive)': aa_max_alive,
        'AA min (alive)': aa_min_alive,
        'BB max (alive)': bb_max_alive,
        'BB min (alive)': bb_min_alive,
    }
    
    # Add alive-dead correlations if dead components exist
    # if display_dead:
    #     correlation_dict['AA max alive→dead'] = aa_max_alive_dead
    #     correlation_dict['BB max alive→dead'] = bb_max_alive_dead
    #     correlation_dict['AA min alive→dead'] = aa_min_alive_dead
    #     correlation_dict['BB min alive→dead'] = bb_min_alive_dead
    
    # Create DataFrame for alive components
    correlation_data = pd.DataFrame(correlation_dict)
    
    # If dead components exist, we need to create a custom correlation matrix
    if display_dead:
        # Create separate DataFrames for dead component features
        dead_data = pd.DataFrame({
            'AA max (dead)': aa_max_dead,
            'AA min (dead)': aa_min_dead,
            'BB max (dead)': bb_max_dead,
            'BB min (dead)': bb_min_dead,
            # 'AA max dead→alive': aa_max_dead_alive,
            # 'BB max dead→alive': bb_max_dead_alive,
            # 'AA min dead→alive': aa_min_dead_alive,
            # 'BB min dead→alive': bb_min_dead_alive,
        })
        
        # Calculate correlations
        alive_corr = correlation_data.corr()
        dead_corr = dead_data.corr()
        
        # Create a larger matrix to hold all correlations
        all_features = list(correlation_data.columns) + list(dead_data.columns)
        n_features = len(all_features)
        full_corr_matrix = pd.DataFrame(np.nan, index=all_features, columns=all_features)
        
        # Fill in alive-alive correlations
        full_corr_matrix.loc[alive_corr.index, alive_corr.columns] = alive_corr
        
        # Fill in dead-dead correlations
        full_corr_matrix.loc[dead_corr.index, dead_corr.columns] = dead_corr
        
        # Calculate cross-correlations for specific pairs        # Calculate cross-correlations for specific pairs
        # AA max alive→dead pairs with AA max (alive) - same length
# Calculate cross-correlations for specific pairs
        # Helper function to safely compute correlation
        def safe_corrcoef(x, y):
            """Compute correlation coefficient, returning NaN if either array has zero variance"""
            if np.std(x) == 0 or np.std(y) == 0:
                return np.nan
            return np.corrcoef(x, y)[0, 1]

        full_corr_matrix.loc['AA max (dead)', 'AA max (alive)'] = safe_corrcoef(aa_max_alive_dead, aa_max_alive)
        full_corr_matrix.loc['AA max (dead)', 'AA min (alive)'] = safe_corrcoef(aa_max_alive_dead, aa_min_alive)
        full_corr_matrix.loc['AA min (dead)', 'AA min (alive)'] = safe_corrcoef(aa_min_alive_dead, aa_min_alive)
        full_corr_matrix.loc['AA min (dead)', 'AA max (alive)'] = safe_corrcoef(aa_min_alive_dead, aa_max_alive)

        # Repeat for dead->alive
        full_corr_matrix.loc['AA max (alive)', 'AA max (dead)'] = safe_corrcoef(aa_max_dead_alive, aa_max_dead)
        full_corr_matrix.loc['AA max (alive)', 'AA min (dead)'] = safe_corrcoef(aa_max_dead_alive, aa_min_dead)
        full_corr_matrix.loc['AA min (alive)', 'AA min (dead)'] = safe_corrcoef(aa_min_dead_alive, aa_min_dead)
        full_corr_matrix.loc['AA min (alive)', 'AA max (dead)'] = safe_corrcoef(aa_min_dead_alive, aa_max_dead)

        # # AA max alive→dead pairs with AA max (alive) - same length
        full_corr_matrix.loc['BB max (dead)', 'BB max (alive)'] = safe_corrcoef(bb_max_alive_dead, bb_max_alive)
        full_corr_matrix.loc['BB max (dead)', 'BB min (alive)'] = safe_corrcoef(bb_max_alive_dead, bb_min_alive)
        full_corr_matrix.loc['BB min (dead)', 'BB min (alive)'] = safe_corrcoef(bb_min_alive_dead, bb_min_alive)
        full_corr_matrix.loc['BB min (dead)', 'BB max (alive)'] = safe_corrcoef(bb_min_alive_dead, bb_max_alive)

        # Repeat for dead->alive
        full_corr_matrix.loc['BB max (alive)', 'BB max (dead)'] = safe_corrcoef(bb_max_dead_alive, bb_max_dead)
        full_corr_matrix.loc['BB max (alive)', 'BB min (dead)'] = safe_corrcoef(bb_max_dead_alive, bb_min_dead)
        full_corr_matrix.loc['BB min (alive)', 'BB min (dead)'] = safe_corrcoef(bb_min_dead_alive, bb_min_dead)
        full_corr_matrix.loc['BB min (alive)', 'BB max (dead)'] = safe_corrcoef(bb_min_dead_alive, bb_max_dead)
        correlation_matrix = full_corr_matrix
    else:
        correlation_matrix = correlation_data.corr()
    
    # Create a mask for NaN values
    mask = correlation_matrix.isna()
    
    # Plot correlation matrix
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8}, ax=ax_corr, mask=mask,
                vmin=-1, vmax=1)
    ax_corr.set_title(f'Correlation Matrix - {layer_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    
    # Create histograms for all cosine similarities
    n_histograms = 16 if display_dead else 4
    n_cols = 4
    n_rows = (n_histograms + n_cols - 1) // n_cols
    
    fig_hist, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Prepare histogram data
    hist_data = [
        ('AA max (alive)', aa_max_alive, (0, 1)),
        ('AA min (alive)', aa_min_alive, (-1, 0)),
        ('BB max (alive)', bb_max_alive, (0, 1)),
        ('BB min (alive)', bb_min_alive, (-1, 0)),
    ]
    
    if display_dead:
        hist_data.extend([
            ('AA max (dead)', aa_max_dead, (0, 1)),
            ('AA min (dead)', aa_min_dead, (-1, 0)),
            ('BB max (dead)', bb_max_dead, (0, 1)),
            ('BB min (dead)', bb_min_dead, (-1, 0)),
            ('AA max alive→dead', aa_max_alive_dead, (0, 1)),
            ('BB max alive→dead', bb_max_alive_dead, (0, 1)),
            ('AA max dead→alive', aa_max_dead_alive, (0, 1)),
            ('BB max dead→alive', bb_max_dead_alive, (0, 1)),
            ('AA min alive→dead', aa_min_alive_dead, (-1, 0)),
            ('BB min alive→dead', bb_min_alive_dead, (-1, 0)),
            ('AA min dead→alive', aa_min_dead_alive, (-1, 0)),
            ('BB min dead→alive', bb_min_dead_alive, (-1, 0)),
        ])
    
    for idx, (name, data, xlim) in enumerate(hist_data):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(name)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {name}')
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)
        # set y to log scale
        ax.set_yscale('log')
        
        # Add mean and std statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.text(0.05, 0.95, f'Std: {std_val:.3f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend()
    
    # Hide unused axes
    for idx in range(len(hist_data), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Cosine Similarity Distributions - {layer_name}', fontsize=14)
    plt.tight_layout()
    
    return fig_corr, fig_hist


def visualize_ab_vectors(components, ci,  layer_name, alive_mask, config, perplexity=15, n_iter=1000):
    """
    Create t-SNE visualizations for A and B vectors from a specific layer,
    colored by alive/dead status.
    
    Args:
        components: Dictionary containing component matrices
        causal_importances: Dictionary containing causal importance scores
        layer_name: Name of the layer to visualize
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
        
    Returns:
        fig_dict: Dictionary containing the matplotlib figures
        A_embedded: t-SNE embedded A vectors
        B_embedded: t-SNE embedded B vectors
        labels_A: Labels indicating alive/dead status
    """
    
    # Initialize figure dictionary
    fig_dict = {}
    
    # Get causal importance for this layer
    # ci = causal_importances[layer_name]
    frequencies = (ci >= 0.1).float().mean(dim=(0, 1))
    
    # Get alive and dead masks
    display_dead = alive_mask.sum() > 0 and not config.remove_dead_components
    dead_mask = ~alive_mask
    
    # Get A and B matrices
    A = components[layer_name].A  # [d_model, n_components]
    B = components[layer_name].B  # [n_components, d_out]
    
    # Create figure with two subplots for A and B vectors
    fig_ab, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: t-SNE of A vectors
    # A vectors are columns, so we need to transpose
    A_vectors = A.T.cpu().numpy()  # [n_components, d_model]
    
    # Create labels for coloring
    labels_A = np.where(alive_mask.cpu().numpy(), 'Alive', 'Dead')
    
    # Run t-SNE on A vectors TODO figure out max iter vs n_iter_without_progress
    # tsne_A = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    tsne_A = TSNE(n_components=2, perplexity=perplexity, n_iter_without_progress=500, random_state=42)
    A_embedded = tsne_A.fit_transform(A_vectors)
    
    # Plot A vectors
    if display_dead:
        labels_for_plot = ['Dead', 'Alive']
    else:
        labels_for_plot = ['Alive']
    colors = {'Alive': 'blue', 'Dead': 'red'}
    for label in labels_for_plot:  # Plot dead first so alive points appear on top
        mask = labels_A == label
        ax1.scatter(A_embedded[mask, 0], A_embedded[mask, 1], 
                   label=label, alpha=0.6, s=50, c=colors[label])
    
    ax1.set_title(f't-SNE of A vectors (columns) - {layer_name}', fontsize=14)
    ax1.set_xlabel('t-SNE dimension 1')
    ax1.set_ylabel('t-SNE dimension 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: t-SNE of B vectors
    # B vectors are rows
    B_vectors = B.cpu().numpy()  # [n_components, d_out]
    
    # Run t-SNE on B vectors
    tsne_B = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    B_embedded = tsne_B.fit_transform(B_vectors)
    
    # Plot B vectors with same coloring
    for label in labels_for_plot:  # Plot dead first so alive points appear on top
        mask = labels_A == label
        ax2.scatter(B_embedded[mask, 0], B_embedded[mask, 1], 
                   label=label, alpha=0.6, s=50, c=colors[label])
    
    ax2.set_title(f't-SNE of B vectors (rows) - {layer_name}', fontsize=14)
    ax2.set_xlabel('t-SNE dimension 1')
    ax2.set_ylabel('t-SNE dimension 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add to figure dictionary
    fig_dict[f"tsne/{layer_name}_ab_vectors_{perplexity}"] = fig_ab
    
    # Additional plot: Show frequency distribution for alive components
    if alive_mask.sum() > 0:
        fig_freq, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))
        alive_frequencies = frequencies[alive_mask].cpu().numpy()
        
        # Frequency histogram
        ax3.hist(alive_frequencies, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_title('Frequency Distribution of Alive Components')
        ax3.set_xlabel('Frequency (CI >= 0.1)')
        ax3.set_ylabel('Count')
        
        # Frequency-colored scatter plot
        scatter = ax4.scatter(A_embedded[alive_mask.cpu().numpy(), 0], 
                            A_embedded[alive_mask.cpu().numpy(), 1],
                            c=alive_frequencies, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax4, label='Frequency')
        ax4.set_title('t-SNE of Alive A vectors colored by frequency')
        ax4.set_xlabel('t-SNE dimension 1')
        ax4.set_ylabel('t-SNE dimension 2')
        
        plt.tight_layout()
        
        # Add to figure dictionary
        fig_dict[f"tsne/{layer_name}_alive_frequency_{perplexity}"] = fig_freq
    
    return fig_dict, A_embedded, B_embedded, labels_A



def permute_to_identity(
    ci_vals: Float[Tensor, "batch C"],
) -> tuple[Float[Tensor, "batch C"], Float[Tensor, " C"]]:
    """Permute matrix to make it as close to identity as possible.

    Returns:
        - Permuted mask
        - Permutation indices
    """

    if ci_vals.ndim != 2:
        raise ValueError(f"Mask must have 2 dimensions, got {ci_vals.ndim}")

    batch, C = ci_vals.shape
    effective_rows = min(batch, C)
    perm_indices = torch.zeros(C, dtype=torch.long, device=ci_vals.device)

    perm: list[int] = [0] * C
    used: set[int] = set()
    for i in range(effective_rows):
        sorted_indices: list[int] = torch.argsort(ci_vals[i, :], descending=True).tolist()
        chosen: int = next((col for col in sorted_indices if col not in used), sorted_indices[0])
        perm[i] = chosen
        used.add(chosen)
    remaining: list[int] = sorted(list(set(range(C)) - used))
    for idx, col in enumerate(remaining):
        perm[effective_rows + idx] = col
    new_ci_vals = ci_vals[:, perm]
    perm_indices = torch.tensor(perm, device=ci_vals.device)

    return new_ci_vals, perm_indices


def _plot_causal_importances_figure(
    ci_vals: dict[str, Float[Tensor, "... C"]],
    title_prefix: str,
    colormap: str,
    input_magnitude: float,
    has_pos_dim: bool,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    title_formatter: Callable[[str], str] | None = None,
) -> plt.Figure:
    """Helper function to plot a single mask figure.

    Args:
        ci_vals: Dictionary of causal importances (or causal importances upper leaky relu) to plot
        title_prefix: String to prepend to the title (e.g., "causal importances" or
            "causal importances upper leaky relu")
        colormap: Matplotlib colormap name
        input_magnitude: Input magnitude value for the title
        has_pos_dim: Whether the masks have a position dimension
        orientation: The orientation of the subplots
        title_formatter: Optional callable to format subplot titles. Takes mask_name as input.

    Returns:
        The matplotlib figure
    """
    if orientation == "vertical":
        n_rows, n_cols = len(ci_vals), 1
        figsize = (5, 5 * len(ci_vals))
    else:
        n_rows, n_cols = 1, len(ci_vals)
        figsize = (5 * len(ci_vals), 5)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
        dpi=300,
    )
    axs = np.array(axs)

    images = []
    for j, (mask_name, mask) in enumerate(ci_vals.items()):
        # mask has shape (batch, C) or (batch, pos, C)
        mask_data = mask.detach().cpu().numpy()
        if has_pos_dim:
            assert mask_data.ndim == 3
            mask_data = mask_data[:, 0, :]
        if orientation == "vertical":
            ax = axs[j, 0]
        else:
            ax = axs[0, j]
        im = ax.matshow(mask_data, aspect="auto", cmap=colormap)
        images.append(im)

        # Move x-axis ticks to bottom
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")
        ax.set_xlabel("Subcomponent index")
        ax.set_ylabel("Input feature index")

        # Apply custom title formatting if provided
        title = title_formatter(mask_name) if title_formatter is not None else mask_name
        ax.set_title(title)

    # Add unified colorbar
    norm = plt.Normalize(
        vmin=min(mask.min().item() for mask in ci_vals.values()),
        vmax=max(mask.max().item() for mask in ci_vals.values()),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    # Capitalize first letter of title prefix for the figure title
    fig.suptitle(f"{title_prefix.capitalize()} - Input magnitude: {input_magnitude}")

    return fig


def plot_causal_importance_vals(
    model: ComponentModel,
    components: Mapping[str, LinearComponent | EmbeddingComponent],
    gates: Mapping[str, Gate | GateMLP],
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    input_magnitude: float,
    plot_raw_cis: bool = True,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    title_formatter: Callable[[str], str] | None = None,
) -> tuple[dict[str, plt.Figure], dict[str, Float[Tensor, " C"]]]:
    """Plot the values of the causal importances for a batch of inputs with single active features.

    Args:
        model: The ComponentModel
        components: Dictionary of components
        gates: Dictionary of gates
        batch_shape: Shape of the batch
        device: Device to use
        input_magnitude: Magnitude of input features
        plot_raw_cis: Whether to plot the raw causal importances (blue plots)
        orientation: The orientation of the subplots
        title_formatter: Optional callable to format subplot titles. Takes mask_name as input.

    Returns:
        Tuple of:
            - Dictionary of figures with keys 'causal_importances' (if plot_raw_cis=True) and 'causal_importances_upper_leaky'
            - Dictionary of permutation indices for causal importances
    """
    # First, create a batch of inputs with single active features
    has_pos_dim = len(batch_shape) == 3
    n_features = batch_shape[-1]
    batch = torch.eye(n_features, device=device) * input_magnitude
    if has_pos_dim:
        # NOTE: For now, we only plot the mask of the first pos dim
        batch = batch.unsqueeze(1)

    pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        batch, module_names=list(components.keys())
    )[1]
    As = {module_name: v.A for module_name, v in components.items()}

    ci_raw, ci_upper_leaky_raw = calc_causal_importances(
        pre_weight_acts=pre_weight_acts, As=As, gates=gates, detach_inputs=False
    )

    ci = {}
    ci_upper_leaky = {}
    all_perm_indices = {}

    for k in ci_raw:
        ci[k], _ = permute_to_identity(ci_vals=ci_raw[k])
        ci_upper_leaky[k], all_perm_indices[k] = permute_to_identity(ci_vals=ci_upper_leaky_raw[k])

    # Create figures dictionary
    figures = {}

    if plot_raw_cis:
        ci_fig = _plot_causal_importances_figure(
            ci_vals=ci,
            title_prefix="importance values lower leaky relu",
            colormap="Blues",
            input_magnitude=input_magnitude,
            has_pos_dim=has_pos_dim,
            orientation=orientation,
            title_formatter=title_formatter,
        )
        figures["causal_importances"] = ci_fig

    ci_upper_leaky_fig = _plot_causal_importances_figure(
        ci_vals=ci_upper_leaky,
        title_prefix="importance values",
        colormap="Reds",
        input_magnitude=input_magnitude,
        has_pos_dim=has_pos_dim,
        orientation=orientation,
        title_formatter=title_formatter,
    )
    figures["causal_importances_upper_leaky"] = ci_upper_leaky_fig

    return figures, all_perm_indices


def plot_subnetwork_attributions_statistics(
    mask: Float[Tensor, "batch_size C"],
) -> dict[str, plt.Figure]:
    """Plot a vertical bar chart of the number of active subnetworks over the batch."""
    batch_size = mask.shape[0]
    if mask.ndim != 2:
        raise ValueError(f"Mask must have 2 dimensions, got {mask.ndim}")

    # Sum over subnetworks for each batch entry
    values = mask.sum(dim=1).cpu().detach().numpy()
    bins = list(range(int(values.min().item()), int(values.max().item()) + 2))
    counts, _ = np.histogram(values, bins=bins)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    bars = ax.bar(bins[:-1], counts, align="center", width=0.8)
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels([str(b) for b in bins[:-1]])
    ax.set_ylabel("Count")
    ax.set_xlabel("Number of active subnetworks")
    ax.set_title("Active subnetworks on current batch")

    # Add value annotations on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    fig.suptitle(f"Active subnetworks on current batch (batch_size={batch_size})")
    return {"subnetwork_attributions_statistics": fig}


def plot_matrix(
    ax: plt.Axes,
    matrix: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_format: str = "%.1f",
    norm: plt.Normalize | None = None,
) -> None:
    # Useful to have bigger text for small matrices
    fontsize = 8 if matrix.numel() < 50 else 4
    norm = norm if norm is not None else CenteredNorm()
    im = ax.matshow(matrix.detach().cpu().numpy(), cmap="coolwarm", norm=norm)
    # If less than 500 elements, show the values
    if matrix.numel() < 500:
        for (j, i), label in np.ndenumerate(matrix.detach().cpu().numpy()):
            ax.text(i, j, f"{label:.2f}", ha="center", va="center", fontsize=fontsize)
    ax.set_xlabel(xlabel)
    if ylabel != "":
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.05)
    fig = ax.get_figure()
    assert fig is not None
    fig.colorbar(im, cax=cax, format=tkr.FormatStrFormatter(colorbar_format))
    if ylabel == "Function index":
        n_functions = matrix.shape[0]
        ax.set_yticks(range(n_functions))
        ax.set_yticklabels([f"{L:.0f}" for L in range(1, n_functions + 1)])


def plot_AB_matrices(
    components: dict[str, LinearComponent | EmbeddingComponent],
    all_perm_indices: dict[str, Float[Tensor, " C"]] | None = None,
) -> plt.Figure:
    """Plot A and B matrices for each instance, grouped by layer."""
    As = {k: v.A for k, v in components.items()}
    Bs = {k: v.B for k, v in components.items()}

    n_layers = len(As)

    # Create figure for plotting - 2 rows per layer (A and B)
    fig, axs = plt.subplots(
        2 * n_layers,
        1,
        figsize=(5, 5 * 2 * n_layers),
        constrained_layout=True,
        squeeze=False,
    )
    axs = np.array(axs)

    images = []

    # Plot A and B matrices for each layer
    for j, name in enumerate(sorted(As.keys())):
        # Plot A matrix
        A_data = As[name]
        if all_perm_indices is not None:
            A_data = A_data[:, all_perm_indices[name]]
        A_data = A_data.detach().cpu().numpy()
        im = axs[2 * j, 0].matshow(A_data, aspect="auto", cmap="coolwarm")
        axs[2 * j, 0].set_ylabel("d_in index")
        axs[2 * j, 0].set_xlabel("Component index")
        axs[2 * j, 0].set_title(f"{name} (A matrix)")
        images.append(im)

        # Plot B matrix
        B_data = Bs[name]
        if all_perm_indices is not None:
            B_data = B_data[all_perm_indices[name], :]
        B_data = B_data.detach().cpu().numpy()
        im = axs[2 * j + 1, 0].matshow(B_data, aspect="auto", cmap="coolwarm")
        axs[2 * j + 1, 0].set_ylabel("Component index")
        axs[2 * j + 1, 0].set_xlabel("d_out index")
        axs[2 * j + 1, 0].set_title(f"{name} (B matrix)")
        images.append(im)

    # Add unified colorbar
    all_matrices = list(As.values()) + list(Bs.values())
    norm = plt.Normalize(
        vmin=min(M.min().item() for M in all_matrices),
        vmax=max(M.max().item() for M in all_matrices),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())
    return fig


def create_embed_ci_sample_table(
    causal_importances: dict[str, Float[Tensor, "... C"]],
) -> wandb.Table | None:
    """Create a wandb table visualizing embedding mask values.

    Args:
        causal_importances: Dictionary of causal importances for each component.

    Returns:
        A wandb Table object or None if transformer.wte not in causal_importances.
    """
    if "transformer.wte" not in causal_importances:
        return None

    # Create a 20x10 table for wandb
    table_data = []
    # Add "Row Name" as the first column
    component_names = ["TokenSample"] + ["CompVal" for _ in range(10)]

    for i, ci in enumerate(causal_importances["transformer.wte"][0, :20]):
        active_values = ci[ci > 0.1].tolist()
        # Cap at 10 components
        active_values = active_values[:10]
        formatted_values = [f"{val:.2f}" for val in active_values]
        # Pad with empty strings if fewer than 10 components
        while len(formatted_values) < 10:
            formatted_values.append("0")
        # Add row name as the first element
        table_data.append([f"{i}"] + formatted_values)

    return wandb.Table(data=table_data, columns=component_names)


def plot_mean_component_activation_counts(
    mean_component_activation_counts: dict[str, Float[Tensor, " C"]],
) -> plt.Figure:
    """Plots the mean activation counts for each component module in a grid."""
    n_modules = len(mean_component_activation_counts)
    max_cols = 6
    n_cols = min(n_modules, max_cols)
    # Calculate the number of rows needed, rounding up
    n_rows = math.ceil(n_modules / n_cols)

    # Create a figure with the calculated number of rows and columns
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    # Ensure axs is always a 2D array for consistent indexing, even if n_modules is 1
    axs = axs.flatten()  # Flatten the axes array for easy iteration

    # Iterate through modules and plot each histogram on its corresponding axis
    for i, (module_name, counts) in enumerate(mean_component_activation_counts.items()):
        ax = axs[i]
        try:
            ax.hist(counts.detach().cpu().numpy(), bins=100)
        except ValueError:
            ax.hist(counts.detach().cpu().numpy(), bins=10)  # Fallback for small counts
        ax.set_yscale("log")
        ax.set_title(module_name)  # Add module name as title to each subplot
        ax.set_xlabel("Mean Activation Count")
        ax.set_ylabel("Frequency")

    # Hide any unused subplots if the grid isn't perfectly filled
    for i in range(n_modules, n_rows * n_cols):
        axs[i].axis("off")

    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout()

    return fig


def plot_ci_histograms(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    bins: int = 100,
) -> dict[str, plt.Figure]:
    """Plot histograms of mask values for each layer.

    Args:
        causal_importances: Dictionary of causal importances for each component.
        bins: Number of bins for the histogram.

    Returns:
        Dictionary mapping layer names to histogram figures.
    """
    fig_dict = {}

    for layer_name_raw, layer_ci in causal_importances.items():
        layer_name = layer_name_raw.replace(".", "_")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(layer_ci.flatten().cpu().numpy(), bins=bins)
        ax.set_title(f"Causal importances for {layer_name}")
        ax.set_xlabel("Causal importance value")
        # Use a log scale
        ax.set_yscale("log")
        ax.set_ylabel("Frequency")

        fig_dict[f"mask_vals_{layer_name}"] = fig

    return fig_dict


def create_toy_model_plot_results(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, Gate | GateMLP],
    batch_shape: tuple[int, ...],
    device: str | torch.device,
    **_,
) -> dict[str, plt.Figure]:
    """Create standard plotting results for decomposition experiments.

    This function is used by both resid_mlp and tms experiments to generate
    mask value plots and AB matrix plots.

    Args:
        model: The ComponentModel
        components: Dictionary of components
        gates: Dictionary of gates
        batch_shape: Shape of the batch
        device: Device to use
        **_: Additional keyword arguments (ignored)

    Returns:
        Dictionary of figures
    """
    fig_dict = {}

    figures, all_perm_indices = plot_causal_importance_vals(
        model=model,
        components=components,
        gates=gates,
        batch_shape=batch_shape,
        device=device,
        input_magnitude=0.75,
    )

    # Merge the figures dict into fig_dict
    fig_dict.update(figures)

    fig_dict["AB_matrices"] = plot_AB_matrices(
        components=components, all_perm_indices=all_perm_indices
    )
    return fig_dict
    
def plot_causal_importance_feature_frequencies(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    thresholds: list[float] = [0.1, 1.0],
    n_samples: int = 5000,
) -> dict[str, plt.Figure]:
    """
    Plot histogram of feature activation frequencies for causal importances.
    
    Args:
        causal_importances: Dictionary of tensors, each of shape (batch, sequence, C)
        thresholds: List of threshold values to plot
        n_samples: Number of samples to use for frequency calculation
    
    Returns:
        dict: Dictionary of figure names to matplotlib figures
    """
    
    fig_dict = {}
    
    # Process each layer in the dictionary
    for layer_name_raw, layer_ci in causal_importances.items():
        layer_name = layer_name_raw.replace(".", "_")
        
        # Convert to tensor if needed
        if isinstance(layer_ci, np.ndarray):
            layer_ci = torch.from_numpy(layer_ci)
        
        # Handle sample limiting
        actual_samples = min(n_samples, layer_ci.shape[0])
        ci_subset = layer_ci[:actual_samples]  # Shape: (batch, seq, C)
        
        for threshold in thresholds:
            # Vectorized calculation of activation frequencies
            # Shape: (batch, seq, C) -> (C,)
            # Calculate fraction of positions where each feature > threshold
            activation_frequencies = (ci_subset >= threshold).float().mean(dim=(0, 1)).cpu().numpy()
            
            # Handle case where all frequencies might be zero
            non_zero_freqs = activation_frequencies[activation_frequencies > 0]
            if len(non_zero_freqs) == 0:
                min_freq = 1e-6
            else:
                min_freq = non_zero_freqs.min()
            
            # Create log-spaced bins
            log_bins = np.logspace(np.log10(min_freq/2), np.log10(1.0), 20)
            
            # Create the histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            counts, bins, patches = ax.hist(activation_frequencies, bins=log_bins, 
                                           edgecolor='black', alpha=0.7)
            
            # Set log scale for x-axis
            ax.set_xscale('log')
            ax.set_xlabel('Causal Importance Activation Frequency')
            ax.set_ylabel('Number of Components')
            ax.set_title(f'Distribution of Causal Importance Values > {threshold} for {layer_name}')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_freq = activation_frequencies.mean()
            median_freq = np.median(activation_frequencies)
            ax.axvline(mean_freq, color='red', linestyle='--', label=f'Mean: {mean_freq:.4f}')
            ax.axvline(median_freq, color='orange', linestyle='--', label=f'Median: {median_freq:.4f}')
            ax.legend()
            
            plt.tight_layout()
            
            # Add to figure dictionary with descriptive name
            threshold_str = str(threshold).replace('.', '_')
            fig_dict[f'ci_feature_freq_{layer_name}_threshold_{threshold_str}'] = fig
            plt.close(fig)  # Close to prevent memory issues
    
    return fig_dict