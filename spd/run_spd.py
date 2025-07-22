"""Run SPD on a model."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from regex import T
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config
from spd.log import logger
from spd.losses import (
    calc_accuracies,
    calc_ce_losses,
    calculate_losses,
)
from spd.models.component_model import ComponentModel, init_As_and_Bs_
from spd.models.component_utils import (
    calc_causal_importances,
    calc_ci_l_zero,
    component_activation_statistics,
    calc_stochastic_masks,
)
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.plotting import (
    create_embed_ci_sample_table,
    plot_ci_histograms,
    plot_mean_component_activation_counts,
    plot_causal_importance_feature_frequencies,
    visualize_ab_vectors,
    plot_all_cos_sims,
)
from spd.utils import (
    calc_kl_divergence_lm,
    calc_mean_squared_error,
    extract_batch_data,
    get_lr_schedule_fn,
    get_lr_with_warmup,
    get_pnorm_schedule_fn,
)

TASK_TO_INPUT_KEY = {
    "lm": "input_ids",
    "cv": "pixel_values",
}

def optim_scale_fn(
    target_weights: Float[Tensor, ""],
    optimizer: torch.optim.Optimizer,
    key: str = "square_avg",
    normalize: bool = True,
):
    square_avg = optimizer.state[target_weights][key]
    if normalize:
        square_avg = square_avg / (square_avg.mean() + 1e-8)
    return square_avg


def get_common_run_name_suffix(config: Config) -> str:
    """Generate a run suffix based on Config that is common to all experiments."""
    run_suffix = ""
    run_suffix += f"nmasks{config.n_mask_samples}_"
    if config.stochastic_recon_coeff is not None:
        run_suffix += f"stochrecon{config.stochastic_recon_coeff:.2e}_"
    if config.stochastic_recon_layerwise_coeff is not None:
        run_suffix += f"stochreconlayer{config.stochastic_recon_layerwise_coeff:.2e}_"
    if config.schatten_coeff is not None:
        run_suffix += f"schatten{config.schatten_coeff:.2e}_"
    if config.embedding_recon_coeff is not None:
        run_suffix += f"embedrecon{config.embedding_recon_coeff:.2e}_"
    run_suffix += f"p{config.pnorm:.2e}_"
    run_suffix += f"impmin{config.importance_minimality_coeff:.2e}_"
    run_suffix += f"C{config.C}_"
    run_suffix += f"sd{config.seed}_"
    run_suffix += f"lr{config.lr:.2e}_"
    run_suffix += f"bs{config.batch_size}_"
    run_suffix += f"{config.faithfulness_scale or ""}importance_"
    run_suffix += f"{config.gate_config.gate_type}_"
    return run_suffix


def optimize(
    target_model: nn.Module,
    config: Config,
    device: str,
    train_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    eval_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_eval_steps: int,
    out_dir: Path | None,
    plot_results_fn: Callable[..., dict[str, plt.Figure]] | None = None,
    tied_weights: list[tuple[str, str]] | None = None,
) -> None:
    """Run the optimization loop for LM decomposition."""

    model = ComponentModel(
        base_model=target_model,
        target_module_patterns=config.target_module_patterns,
        C=config.C,
        gate_config=config.gate_config,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        filler_comp_bool=config.learned_filler_comp,
    )

    for param in target_model.parameters():
        param.requires_grad = False
    logger.info("Target model parameters frozen.")

    faithfulness_no_scale_fn = lambda x : torch.tensor(1.0, device=x.device if hasattr(x, "device") else "cpu")
    if config.faithfulness_scale == "rms":
        non_bias_params = [
            p for name, p in target_model.named_parameters() if "bias" not in name
        ]
        for p in non_bias_params:
            p.requires_grad = True
        rms_opt = torch.optim.RMSprop(
            non_bias_params,
            lr=0.0
        )
        faithfulness_scale_fn = partial(
            optim_scale_fn,
            key="square_avg",
            optimizer=rms_opt,
            normalize=True,
        )
    else:
        rms_opt = None
        faithfulness_scale_fn = faithfulness_no_scale_fn

    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in model.gates.items()
    }  # type: ignore
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in model.components.items()
    }  # type: ignore

    model.to(device)
    init_As_and_Bs_(model=model, components=components)

    if tied_weights is not None:
        # Tie component weights. Assume that the first element is a transpose of the second element
        # NOTE: Tying weights will make your training nondeterministic
        for src_name, tgt_name in tied_weights:
            components[tgt_name].B.data = components[src_name].A.data.T
            components[tgt_name].A.data = components[src_name].B.data.T


    # Initialize the filler comp weight to be the residual of the weight
    if config.learned_filler_comp:
        if(config.init_filler_comp_to_residual):
            with torch.no_grad():
                for comp_name, component in model.components.items():
                    component_params = component.weight

                    submodule = target_model.get_submodule(comp_name.replace("-", "."))
                    assert isinstance(submodule, nn.Linear | nn.Embedding)
                    target_params = submodule.weight

                    diff = target_params - component_params
                    # initialize the filler comp weight to be the residual of the weight
                    component.filler_comp_weight.data = diff.T

    component_params: list[torch.nn.Parameter] = []
    gate_params: list[torch.nn.Parameter] = []
    for _, component in components.items():
        component_params.extend(list(component.parameters()))
    for _, gate in gates.items():
        gate_params.extend(gate.parameters())

    assert len(component_params) > 0, "No parameters found in components to optimize"

    optimizer = optim.AdamW(component_params + gate_params, lr=config.lr, weight_decay=0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)
    logger.info(f"Base LR scheduler created: {config.lr_schedule}")

    # Or create different scheduler configurations
    if config.pannealing:
        cosine_scheduler = get_pnorm_schedule_fn(
            start_value=config.pnorm,
            end_value=config.pnorm_min,
            warmup_fraction=config.p_anneal_warmup_pct,
            schedule_type=config.p_anneal_schedule_type,
            total_steps=config.p_anneal_steps or config.steps,
        )

    n_params = sum(model.model.get_parameter(n + ".weight").numel() for n in components)

    log_data = {}
    data_iter = iter(train_loader)

    alive_components: dict[str, Bool[Tensor, " C"]] = {
        layer_name: torch.zeros(config.C, device=device).bool() for layer_name in components
    }

    input_key = TASK_TO_INPUT_KEY[config.task_config.task_name]

    # Iterate one extra step for final logging/plotting/saving
    for step in tqdm(range(config.steps + 1), ncols=0):
        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )
        if config.pannealing:
            pnorm = cosine_scheduler(step) # type: ignore
        else:
            pnorm = config.pnorm

        for group in optimizer.param_groups:
            group["lr"] = step_lr
        log_data["lr"] = step_lr

        optimizer.zero_grad()

        try:
            batch_item = next(data_iter)
            batch, labels = extract_batch_data(batch_item, input_key=input_key)
        except StopIteration:
            logger.warning("Dataloader exhausted, resetting iterator.")
            data_iter = iter(train_loader)
            batch_item = next(data_iter)
            batch, labels = extract_batch_data(batch_item, input_key=input_key)
        batch = batch.to(device)
        if labels is not None:
            labels = labels.to(device)

        target_out, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        As = {module_name: components[module_name].A for module_name in components}

        causal_importances, causal_importances_upper_leaky = calc_causal_importances(
            pre_weight_acts=pre_weight_acts, As=As, gates=gates, detach_inputs=False
        )

        for layer_name, ci in causal_importances.items():
            alive_components[layer_name] = alive_components[layer_name] | (ci > 0.1).any(dim=(0, 1))

        total_loss, loss_terms = calculate_losses(
            model=model,
            batch=batch,
            config=config,
            pnorm=pnorm,
            components=components,
            causal_importances=causal_importances,
            causal_importances_upper_leaky=causal_importances_upper_leaky,
            target_out=target_out,
            device=device,
            n_params=n_params,
            faithfulness_scale_fn=faithfulness_scale_fn if step > config.lr_warmup_pct * config.steps and step > 1  else faithfulness_no_scale_fn,
        )

        log_data["loss/total"] = total_loss.item()
        log_data.update(loss_terms)

        with torch.inference_mode():
            if step > 0 and step % config.check_dead_components_freq == 0:
                dead_fig_dict = {}
                for layer_name, layer_alive_components in alive_components.items():
                    log_data[f"{layer_name}/n_alive_01"] = layer_alive_components.sum().item()
                    # I think we want to remove components that are not alive (or very infrequent)
                    # dead_components = ~layer_alive_components
                    # I'm unsure how to remove components in general. Maybe set to zero, set requires_grad to false? 
                    # For computation sake, it'd be better to just remove it, but will mess up indexing for other ones. 
                    # Maybe not since we're zero-ing out the alive components afterwards
                    if config.remove_dead_components:
                        dead_components = ~layer_alive_components
                        num_removed = dead_components.sum().item()
                        # print(f"\nRemoving {num_removed} components from {layer_name}")
                        components[layer_name].A.data[:, dead_components] = 0
                        components[layer_name].B.data[dead_components, :] = 0
                        # clear the current grad
                        log_data[f"{layer_name}/removed_components"] = num_removed

                # for layer_name, ci in causal_importances.items():
                    # ci = causal_importances[layer_name]
                    # Needs to be .items() related
                    alive_mask = layer_alive_components 

                    fig_corr, fig_hist = plot_all_cos_sims(components, ci, layer_name, alive_mask, config, log_data)
                    dead_fig_dict[f"cos_sim/{layer_name}_correlation_matrix_full"] = fig_corr
                    dead_fig_dict[f"cos_sim/{layer_name}_cosine_similarity_histograms_full"] = fig_hist

                    # Also do TNSE
                    # for perplexity in [10, 15, 20, 25]:
                    for perplexity in [10, 25, 40]:
                        print(f"Doing TNSE for {layer_name}")
                        layer_figs, _, _, _ = visualize_ab_vectors(components, ci, layer_name, alive_mask=alive_mask, config=config, perplexity=perplexity, n_iter=5000)
                        print(f"TNSE done for {layer_name}")
                        dead_fig_dict.update(layer_figs)
                    alive_components[layer_name] = torch.zeros(config.C, device=device).bool()
                
                if config.wandb_project:
                    wandb.log(
                        {k: wandb.Image(v) for k, v in dead_fig_dict.items()},
                        step=step,
                    )

            # --- Logging --- #
            if step % config.print_freq == 0:
                tqdm.write(f"--- Step {step} ---")
                tqdm.write(f"LR: {step_lr:.6f}")
                tqdm.write(f"PNorm: {pnorm:.6f}")
                tqdm.write(f"Total Loss: {log_data['loss/total']:.7f}")
                for name, value in loss_terms.items():
                    tqdm.write(f"{name}: {value:.7f}")

                # TODO Replace w/ a function that calculate both KL, CE-diff, etc, just given the masks
                # Would need to initially compute the CE & Logits for original model

                
                # Calculate component logits and KL losses
                masked_component_logits = model.forward_with_components(
                    batch, components=components, masks=causal_importances, filler_comp_scalar=0.0,
                )
                ones_masks = {k: torch.ones_like(v) for k, v in causal_importances.items()}
                all_components_logits = model.forward_with_components(
                    batch, components=components, masks=ones_masks, filler_comp_scalar=1.0,
                )

                
                # stochastic
                stochastic_masks = calc_stochastic_masks(
                    causal_importances=causal_importances, n_mask_samples=config.n_mask_samples
                )[0]
                stochastic_component_logits = model.forward_with_components(
                    batch, components=components, masks=stochastic_masks, filler_comp_scalar=0.0,
                )

                target_logits = model(batch)

                log_data["misc/all_components_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=all_components_logits, target=target_logits
                ).item()
                log_data["misc/masked_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=masked_component_logits, target=target_logits
                ).item()
                log_data["misc/stochastic_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=stochastic_component_logits, target=target_logits
                ).item()
                # clamped_mask
                clamped_masks = {k: v.clamp(0, 1) for k, v in causal_importances.items()}
                clamped_masked_component_logits = model.forward_with_components(
                    batch, components=components, masks=clamped_masks
                )
                log_data["misc/clamped_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=clamped_masked_component_logits, target=target_logits
                ).item()
                if config.learned_filler_comp:
                    # Evaluate Clamped loss w/ & w/o Filler added   
                    clamped_masked_component_logits_no_filler = model.forward_with_components(
                        batch, components=components, masks=clamped_masks, filler_comp_scalar=0.0
                    )
                    clamped_masked_component_logits_filler = model.forward_with_components(
                        batch, components=components, masks=clamped_masks, filler_comp_scalar=1.0,
                    )
                    clamped_masked_component_logits_filler_max = model.forward_with_components(
                        batch, components=components, masks=clamped_masks, filler_comp_scalar=config.max_filler_scalar,
                    )
                    clamped_no_filler_kl_loss = calc_kl_divergence_lm(
                        pred=clamped_masked_component_logits_no_filler, target=target_logits
                    ).item()
                    clamped_filler_kl_loss = calc_kl_divergence_lm(
                        pred=clamped_masked_component_logits_filler, target=target_logits
                    ).item()
                    clamped_filler_max_kl_loss = calc_kl_divergence_lm(
                        pred=clamped_masked_component_logits_filler_max, target=target_logits
                    ).item()
                    clamped_filler_diff =  clamped_filler_kl_loss - clamped_no_filler_kl_loss
                    log_data["filler/clamped_kl_NO_filler"] = clamped_no_filler_kl_loss
                    log_data["filler/clamped_kl_WITH_filler"] = clamped_filler_kl_loss
                    log_data["filler/clamped_kl_diff_(~0 is better)"] = clamped_filler_diff
                    log_data["filler/clamped_kl_WITH_filler_MAX"] = clamped_filler_max_kl_loss
                    log_data["filler/clamped_kl_WITH_filler_MAX_diff_(~0 is better)"] = clamped_filler_max_kl_loss - clamped_no_filler_kl_loss


                    # Repeat for 1's mask
                    ones_masked_component_logits_no_filler = model.forward_with_components(
                        batch, components=components, masks=ones_masks, filler_comp_scalar=0.0
                    )
                    ones_masked_component_logits_filler = model.forward_with_components(
                        batch, components=components, masks=ones_masks, filler_comp_scalar=1.0,
                    )
                    ones_no_filler_kl_loss = calc_kl_divergence_lm(
                        pred=ones_masked_component_logits_no_filler, target=target_logits
                    ).item()
                    ones_filler_kl_loss = calc_kl_divergence_lm(
                        pred=ones_masked_component_logits_filler, target=target_logits
                    ).item()
                    ones_filler_diff = ones_no_filler_kl_loss - ones_filler_kl_loss
                    log_data["filler/ones_kl_NO_filler"] = ones_no_filler_kl_loss
                    log_data["filler/ones_kl_WITH_filler"] = ones_filler_kl_loss
                    log_data["filler/ones_kl_diff_(higher_is_better)"] = ones_filler_diff


                    for comp_name, component in model.components.items():
                        component_params = component.weight

                        submodule = target_model.get_submodule(comp_name.replace("-", "."))
                        assert isinstance(submodule, nn.Linear | nn.Embedding)
                        target_params = submodule.weight
                # submodule = target_model.get_submodule(comp_name.replace("-", "."))

                        no_filler_diff = ((target_params - component_params)**2).sum() / component_params.numel()
                        filler_diff = ((target_params - (component_params + component.filler_comp_weight.T))**2).sum() / component_params.numel()  
                        log_data[f"filler/faithfulness{comp_name}_no_filler"] = no_filler_diff
                        log_data[f"filler/faithfulness{comp_name}_filler"] = filler_diff
                        log_data[f"filler/faithfulness{comp_name}_filler_diff_(higher_is_better)"] =  no_filler_diff - filler_diff

                    # log_data["filler/ones_kl_loss_vs_target_no_filler"] = calc_kl_divergence_lm(
                    #     pred=ones_masked_component_logits_no_filler, target=target_logits
                    # ).item()
                    # log_data["filler/ones_kl_loss_vs_target_filler"] = calc_kl_divergence_lm(
                    #     pred=ones_masked_component_logits_filler, target=target_logits
                    # ).item()


                    # Also log the norm of the filler_comp for our linear cmoponent 
                    for layer_name, component in components.items():
                        if component.filler_comp_bool:
                            log_data[f"filler/norm_filler_comp_{layer_name}"] = component.filler_comp_weight.norm().item()
                # binned mask (do bin_num = 10)
                bin_num = 5
                # Bin the masks to nearest fraction (e.g., bin_num=10 -> 0.0, 0.1, 0.2, ..., 1.0)
                binned_masks = {}
                for k, v in causal_importances.items():
                    # First clamp to [0, 1] range
                    clamped = v.clamp(0, 1)
                    # Round to nearest bin
                    binned = torch.round(clamped * bin_num) / bin_num
                    binned_masks[k] = binned

                binned_masked_component_logits = model.forward_with_components(
                    batch, components=components, masks=binned_masks
                )
                log_data["misc/binned_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=binned_masked_component_logits, target=target_logits
                ).item()

                log_data["misc/all_components_mse_vs_target"] = calc_mean_squared_error(
                    pred=all_components_logits, target=target_logits
                ).item()
                log_data["misc/masked_mse_vs_target"] = calc_mean_squared_error(
                    pred=masked_component_logits, target=target_logits
                ).item()
                log_data["misc/stochastic_mse_vs_target"] = calc_mean_squared_error(
                    pred=stochastic_component_logits, target=target_logits
                ).item()
                log_data["misc/clamped_mse_vs_target"] = calc_mean_squared_error(
                    pred=clamped_masked_component_logits, target=target_logits
                ).item()
                noisy_masks = {k: torch.randn_like(v) * config.noise_log_std for k, v in causal_importances.items()}
                noisy_masked_component_logits = model.forward_with_components(
                    batch, components=components, masks=noisy_masks
                )
                log_data["misc/noisy_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=noisy_masked_component_logits, target=target_logits
                ).item()

                if config.log_ce_losses:
                    ce_losses = calc_ce_losses(
                        model=model,
                        batch=batch,
                        components=components,
                        masks=causal_importances,
                        all_components_logits=all_components_logits,
                        masked_component_logits=masked_component_logits,
                        clamped_masked_component_logits=clamped_masked_component_logits,
                        stochastic_component_logits=stochastic_component_logits,
                        binned_masked_component_logits=binned_masked_component_logits,
                        noisy_masked_component_logits=noisy_masked_component_logits,
                        target_logits=target_logits,
                        task=config.task_config.task_name,  # type: ignore[call-arg]
                        labels=labels,
                    )
                    log_data.update(ce_losses)

                if config.log_accuracies:
                    if config.task_config.task_name in ["lm", "cv"]:
                        acc = calc_accuracies(
                            model=model,
                            batch=batch,
                            components=components,
                            masks=causal_importances,
                            all_components_logits=all_components_logits,
                            masked_component_logits=masked_component_logits,
                            stochastic_component_logits=stochastic_component_logits,
                            clamped_masked_component_logits=clamped_masked_component_logits,
                            binned_masked_component_logits=binned_masked_component_logits,
                            noisy_masked_component_logits=noisy_masked_component_logits,
                            target_logits=target_logits,
                            task=config.task_config.task_name, # type: ignore[call-arg]
                            labels=labels,
                        )
                        log_data.update(acc)

                embed_ci_table = create_embed_ci_sample_table(causal_importances)
                if embed_ci_table is not None:
                    log_data["misc/embed_ci_sample"] = embed_ci_table

                if config.wandb_project:
                    ci_l_zero = calc_ci_l_zero(causal_importances=causal_importances)
                    for layer_name, layer_ci_l_zero in ci_l_zero.items():
                        log_data[f"{layer_name}/ci_l0"] = layer_ci_l_zero
                    wandb.log(log_data, step=step)

            # --- Plotting --- #
            if (
                config.image_freq is not None
                and step % config.image_freq == 0
                and (step > 0 or config.image_on_first_step)
            ):
                logger.info(f"Step {step}: Generating plots...")
                fig_dict = {}
                if plot_results_fn is not None:
                    fig_dict = plot_results_fn(
                        model=model,
                        components=components,
                        gates=gates,
                        batch_shape=batch.shape,
                        device=device,
                    )

                ci_histogram_figs = plot_ci_histograms(causal_importances=causal_importances)
                fig_dict.update(ci_histogram_figs)

                # Add the new feature frequency histograms
                ci_feature_freq_figs = plot_causal_importance_feature_frequencies(
                    causal_importances=causal_importances,
                    thresholds=[0.1, 1.0],
                    n_samples=5000
                )
                fig_dict.update(ci_feature_freq_figs)

        
                # # Now check cos-sim of A & B. We want AxA & BxB and AxB
                # # Would be great to color by frequency
                # for layer_name, ci in causal_importances.items():
                #     frequencies = (ci >= 0.1).float().mean(dim=(0, 1))
                #     # dead are set to zero
                #     # dead_mask = components[layer_name].A.sum(dim=0) == 0
                #     alive_mask = components[layer_name].A.sum(dim=0) > 0
                #     A_alive = components[layer_name].A[:, alive_mask]
                #     B_alive = components[layer_name].B[alive_mask, :]
                #     alive_frequencies = frequencies[alive_mask].cpu().numpy()
                #     print(f" Freq shape: {frequencies.shape}, alive_frequencies shape: {alive_frequencies.shape}")
                #     print(f" a alive shape: {A_alive.shape}, b alive shape: {B_alive.shape}")
                #     print(f" alive mask shape: {alive_mask.shape}")
                #     print(f" Num Alive: {alive_mask.sum().item()}")

                #     # Now we want to do the same for the dead components
                #     normed_A = torch.nn.functional.normalize(A_alive, dim=0)  # [d_model, n_components]
                #     normed_B = torch.nn.functional.normalize(B_alive, dim=1)  # [n_components, d_out]
                #     aa_cos_sim = (normed_A.T @ normed_A).tril(diagonal=-1) # shape (C, C)
                #     bb_cos_sim = (normed_B @ normed_B.T).tril(diagonal=-1) # shape (C, C)
                #     print(f" aa cos sim shape: {aa_cos_sim.shape}, bb cos sim shape: {bb_cos_sim.shape}")
                    
                #     # Find the max & min for each comp
                #     aa_max = aa_cos_sim.max(dim=0).values
                #     aa_min = aa_cos_sim.min(dim=0).values
                #     bb_max = bb_cos_sim.max(dim=0).values
                #     bb_min = bb_cos_sim.min(dim=0).values

                #     log_data[f"cos_sim/{layer_name}_aa_max_mean"] = aa_max.mean().item()
                #     log_data[f"cos_sim/{layer_name}_aa_min_mean"] = aa_min.mean().item()
                #     log_data[f"cos_sim/{layer_name}_bb_max_mean"] = bb_max.mean().item()
                #     log_data[f"cos_sim/{layer_name}_bb_min_mean"] = bb_min.mean().item()
                #     aa_max = aa_max.cpu().numpy()
                #     aa_min = aa_min.cpu().numpy()
                #     bb_max = bb_max.cpu().numpy()
                #     bb_min = bb_min.cpu().numpy()


                #     # Now we want the scatter plot of all combos 
                #     # Let's create names so we can loop & do it automatically, w/ below as example
                #     cos_sim_names = ["aa_max", "aa_min", "bb_max", "bb_min"]
                #     temp_fig_dict = {}
                #     for i in range(len(cos_sim_names)):
                #         for j in range(i+1, len(cos_sim_names)):
                #             cos_sim_name_i = cos_sim_names[i]
                #             cos_sim_name_j = cos_sim_names[j]
                #             fig, ax = plt.subplots(figsize=(10, 6))

                #             # Get the appropriate data based on the names
                #             data_i = locals()[cos_sim_name_i]  # This gets aa_max, aa_min, etc.
                #             data_j = locals()[cos_sim_name_j]

                #             # Set colorbar limits based on whether it's min or max
                #             if "min" in cos_sim_name_j:
                #                 vmin, vmax = -1, 0
                #             else:
                #                 vmin, vmax = 0, 1
                            
                #             scatter = ax.scatter(alive_frequencies, data_i, c=data_j, cmap="viridis", alpha=0.5, 
                #                             vmin=vmin, vmax=vmax)         
                #             # Add colorbar with label
                #             cbar = plt.colorbar(scatter, ax=ax)
                #             cbar.set_label(f"{cos_sim_name_j}", rotation=270, labelpad=20)
                            
                #             ax.set_xlabel("Frequency (log scale)")
                #             ax.set_ylabel(f"{cos_sim_name_i}")
                #             ax.set_xscale('log')  # Set x-axis to log scale
                            
                #             # Set y limits based on whether it's min or max
                #             if "min" in cos_sim_name_i:
                #                 ax.set_ylim(-1, 0)
                #             else:
                #                 ax.set_ylim(0, 1)
                                
                #             ax.set_title(f"Frequency vs {cos_sim_name_i} (colored by {cos_sim_name_j})")
                            
                #             temp_fig_dict[f"cos_sim/{layer_name}_freq_vs_{cos_sim_name_i}_colored_by_{cos_sim_name_j}"] = fig
                            
                #     fig_dict.update(temp_fig_dict)

          


                mean_component_activation_counts = component_activation_statistics(
                    model=model, dataloader=eval_loader, n_steps=n_eval_steps, device=device, input_key=input_key
                )[1]
                assert mean_component_activation_counts is not None
                fig_dict["mean_component_activation_counts"] = (
                    plot_mean_component_activation_counts(
                        mean_component_activation_counts=mean_component_activation_counts,
                    )
                )

                if config.wandb_project:
                    wandb.log(
                        {k: wandb.Image(v) for k, v in fig_dict.items()},
                        step=step,
                    )
                    # if out_dir is not None:
                    #     for k, v in fig_dict.items():
                    #         v.savefig(out_dir / f"{k}_{step}.png")
                    #         tqdm.write(f"Saved plot to {out_dir / f'{k}_{step}.png'}")

        # --- Saving Checkpoint --- #
        if (
            (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
            or step == config.steps
        ) and out_dir is not None:
            torch.save(model.state_dict(), out_dir / f"model_{step}.pth")
            logger.info(f"Saved model, optimizer, and out_dir to {out_dir}")
            if config.wandb_project:
                wandb.save(str(out_dir / f"model_{step}.pth"), base_path=str(out_dir), policy="now")
                wandb.save(
                    str(out_dir / f"optimizer_{step}.pth"), base_path=str(out_dir), policy="now"
                )

        # --- Backward Pass & Optimize --- #
        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            total_loss.backward()
            if config.faithfulness_scale == "rms" and rms_opt is not None:
                rms_opt.step()

            target_param_ids = {id(p) for p in target_model.parameters()}

            if step % config.print_freq == 0 and config.wandb_project:
                grad_norm: Float[Tensor, ""] = torch.zeros((), device=device)
                for param in model.parameters():
                    # Do not count the gradients of the target model
                    if id(param) in target_param_ids:
                        continue
                    if param.grad is not None:
                        grad_norm += param.grad.data.flatten().pow(2).sum()  # type: ignore
                grad_norm_val = grad_norm.sqrt().item()
                wandb.log({"grad_norm": grad_norm_val}, step=step)

            optimizer.step()

    logger.info("Finished training loop.")
