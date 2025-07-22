# %%
"""
Example notebook for component analysis using the analysis_utils and plotting_utils modules.
This demonstrates how to analyze dead/alive components in a neural network model.
"""

# %% [markdown]
# # Component Model Analysis
# 
# This notebook demonstrates how to use the component analysis utilities to:
# 1. Calculate component frequencies and importance scores
# 2. Analyze cosine similarities between components
# 3. Visualize the results

# %%
# Imports
import torch
import os
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.data import DatasetConfig, create_data_loader
from transformers import AutoTokenizer

# Import our analysis utilities
from analysis_utils import (
    ComponentAnalysisResults,
    calculate_component_frequency,
    calculate_weight_sim_importance,
    calculate_encoder_decoder_norms,
    calculate_alive_dead_masks,
    calculate_cosine_similarities,
    calculate_all_mmcs_metrics
)
from plotting_utils import (
    plot_encoder_decoder_norms,
    plot_frequency_vs_weight_sim,
    plot_weight_sim_histogram,
    plot_mmcs_histograms,
    plot_alive_frequency_relationships,
    plot_correlation_summary
)

# %%
# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "out/your_model_directory/"
steps = 40000
model_path = os.path.join(model_dir, f"model_{steps}.pth")
config_path = os.path.join(model_dir, f"final_config.yaml")

# Load model
model, config, out_dir = ComponentModel.from_pretrained(model_path)
model.to(device)
model.requires_grad_(False)

# Load tokenizer
model_name = "roneneldan/TinyStories-1M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# %% [markdown]
# ## Setup Data Loader

# %%
# Create evaluation data loader
batch_size = 4
eval_data_config = DatasetConfig(
    name=config.task_config.dataset_name,
    hf_tokenizer_path=config.pretrained_model_name_hf,
    split=config.task_config.eval_data_split,
    n_ctx=config.task_config.max_seq_len,
    is_tokenized=False,
    streaming=False,
    column_name=config.task_config.column_name,
)
eval_loader, _ = create_data_loader(
    dataset_config=eval_data_config,
    batch_size=batch_size,
    buffer_size=config.task_config.buffer_size,
    global_seed=config.seed,
    ddp_rank=0,
    ddp_world_size=1,
)

print(f"Eval dataset name: {eval_data_config.name}")
print(f"Split: {eval_data_config.split}")
print(f"Number of batches: {len(eval_loader)}")

# %% [markdown]
# ## Extract Components and Gates

# %%
# Extract gates and components
module_comp = [key.replace('-', '.') for key in model.components.keys()]

gates = {
    k.removeprefix("gates.").replace("-", "."): v 
    for k, v in model.gates.items()
}

components = {
    k.removeprefix("components.").replace("-", "."): v 
    for k, v in model.components.items()
}

As = {module_name: components[module_name].A for module_name in components}

# Target module for analysis
module_name = config.target_module_patterns[0]
num_components = config.C

print(f"Analyzing module: {module_name}")
print(f"Number of components: {num_components}")

# %% [markdown]
# ## Calculate Component Statistics

# %%
# Initialize results container
results = ComponentAnalysisResults(
    module_name=module_name,
    num_components=num_components
)

# Calculate component frequency
print("Calculating component frequencies...")
results.frequency = calculate_component_frequency(
    model=model,
    eval_loader=eval_loader,
    components=components,
    As=As,
    gates=gates,
    module_comp=module_comp,
    module_name=module_name,
    device=device
)

# Calculate weight similarity importance
print("Calculating weight similarity importance...")
results.weight_sim_importance = calculate_weight_sim_importance(
    components=components,
    model=model,
    num_components=num_components,
    device=device
)

# Calculate encoder/decoder norms
print("Calculating encoder/decoder norms...")
results.enc_norms, results.dec_norms = calculate_encoder_decoder_norms(
    components=components,
    module_name=module_name
)

# Calculate alive/dead masks
results.alive_mask, results.dead_mask = calculate_alive_dead_masks(
    frequency=results.frequency,
    threshold=0
)

print(f"Alive components: {results.alive_mask.sum().item()}")
print(f"Dead components: {results.dead_mask.sum().item()}")

# %% [markdown]
# ## Calculate Cosine Similarities

# %%
# Calculate cosine similarities
print("Calculating cosine similarities...")
results.cosine_similarities = calculate_cosine_similarities(
    components=components,
    module_name=module_name,
    alive_mask=results.alive_mask,
    dead_mask=results.dead_mask
)

# Calculate MMCS metrics
print("Calculating MMCS metrics...")
results.mmcs_metrics = calculate_all_mmcs_metrics(
    components=components,
    module_name=module_name,
    alive_mask=results.alive_mask,
    dead_mask=results.dead_mask
)

# Print summary statistics
print("\nMMCS Summary (Max):")
for key, value in results.mmcs_metrics['max'].items():
    if key.endswith('_mean'):
        print(f"  {key}: {value:.3f}")

print("\nMMCS Summary (Min):")
for key, value in results.mmcs_metrics['min'].items():
    if key.endswith('_mean'):
        print(f"  {key}: {value:.3f}")

# %% [markdown]
# ## Visualizations

# %%
# Plot encoder vs decoder norms
fig1 = plot_encoder_decoder_norms(
    enc_norms=results.enc_norms,
    dec_norms=results.dec_norms,
    alive_mask=results.alive_mask,
    dead_mask=results.dead_mask,
    same_scale=False
)
fig1.show()

# %%
# Plot frequency vs weight sim importance
fig2 = plot_frequency_vs_weight_sim(
    frequency=results.frequency,
    weight_sim_importance=results.weight_sim_importance,
    dec_norms=results.dec_norms
)
fig2.show()

# %%
# Plot weight sim importance histogram
fig3 = plot_weight_sim_histogram(
    weight_sim_importance=results.weight_sim_importance,
    alive_mask=results.alive_mask
)
fig3.show()

# %%
# Plot MMCS histograms
fig4 = plot_mmcs_histograms(
    mmcs_metrics=results.mmcs_metrics,
    plot_type='max'
)
fig4.show()

fig5 = plot_mmcs_histograms(
    mmcs_metrics=results.mmcs_metrics,
    plot_type='min'
)
fig5.show()

# %%
# Plot alive component relationships
alive_freq = results.frequency[results.alive_mask]
alive_enc_norms = results.enc_norms[results.alive_mask]
alive_dec_norms = results.dec_norms[results.alive_mask]
weight_sim_imp_alive = results.weight_sim_importance[results.alive_mask]

fig6 = plot_alive_frequency_relationships(
    alive_freq=alive_freq,
    alive_enc_norms=alive_enc_norms,
    alive_dec_norms=alive_dec_norms,
    mmcs_a_a_enc=results.mmcs_metrics['max']['mmcs_a_a_enc'],
    mmcs_a_a_dec=results.mmcs_metrics['max']['mmcs_a_a_dec'],
    weight_sim_imp_alive=weight_sim_imp_alive
)
fig6.show()

# %% [markdown]
# ## Correlation Analysis

# %%
import numpy as np

# Calculate correlations for alive components
alive_correlations = {
    'Freq vs Enc Norm': np.corrcoef(alive_freq.cpu(), alive_enc_norms.cpu())[0, 1],
    'Freq vs Dec Norm': np.corrcoef(alive_freq.cpu(), alive_dec_norms.cpu())[0, 1],
    'Enc Norm vs Dec Norm': np.corrcoef(alive_enc_norms.cpu(), alive_dec_norms.cpu())[0, 1],
    'Freq vs Weight Sim': np.corrcoef(alive_freq.cpu(), weight_sim_imp_alive.cpu())[0, 1],
    'Freq vs MMCS Enc': np.corrcoef(
        alive_freq.cpu(), 
        results.mmcs_metrics['max']['mmcs_a_a_enc'].cpu()
    )[0, 1],
    'Freq vs MMCS Dec': np.corrcoef(
        alive_freq.cpu(), 
        results.mmcs_metrics['max']['mmcs_a_a_dec'].cpu()
    )[0, 1],
}

fig7 = plot_correlation_summary(
    correlations=alive_correlations,
    title="Alive Component Correlations",
    highlight_threshold=0.5
)
fig7.show()

# %%
# Print detailed statistics
print("Component Analysis Summary")
print("=" * 50)
print(f"Total components: {num_components}")
print(f"Alive components: {results.alive_mask.sum().item()} ({results.alive_mask.sum().item()/num_components*100:.1f}%)")
print(f"Dead components: {results.dead_mask.sum().item()} ({results.dead_mask.sum().item()/num_components*100:.1f}%)")
print(f"\nFrequency range: [{results.frequency.min():.6f}, {results.frequency.max():.6f}]")
print(f"Weight sim range: [{results.weight_sim_importance.min():.3f}, {results.weight_sim_importance.max():.3f}]")
print(f"\nEncoder norm range: [{results.enc_norms.min():.3f}, {results.enc_norms.max():.3f}]")
print(f"Decoder norm range: [{results.dec_norms.min():.3f}, {results.dec_norms.max():.3f}]")

# %%
# Save results for later use
torch.save({
    'results': results,
    'config': config,
    'model_path': model_path
}, 'component_analysis_results.pt')

print("Analysis complete! Results saved to component_analysis_results.pt")