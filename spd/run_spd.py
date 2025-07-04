"""Run SPD on a model."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
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
)
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.plotting import (
    create_embed_ci_sample_table,
    plot_ci_histograms,
    plot_mean_component_activation_counts,
)
from spd.utils import (
    calc_kl_divergence_lm,
    calc_mean_squared_error,
    extract_batch_data,
    get_lr_schedule_fn,
    get_lr_with_warmup,
)

TASK_TO_INPUT_KEY = {
    "lm": "input_ids",
    "cv": "pixel_values",
}

def optim_scale_fn(
    target_weights: Float[Tensor, ""],
    optimizer: torch.optim.Optimizer,
    key: str = "square_avg",
    normalize: bool = False,
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
        n_ci_mlp_neurons=config.n_ci_mlp_neurons,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )

    for param in target_model.parameters():
        param.requires_grad = False
    logger.info("Target model parameters frozen.")

    faithfulness_no_scale_fn = lambda x : torch.tensor(1.0, device=x.device if hasattr(x, "device") else "cpu")
    if config.faithfulness_scale == "rms":
        non_bias_params = [
            p for name, p in model.named_parameters() if "bias" not in name
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

    component_params: list[torch.nn.Parameter] = []
    gate_params: list[torch.nn.Parameter] = []
    for name, component in components.items():
        component_params.extend(list(component.parameters()))
        gate_params.extend(list(gates[name].parameters()))

    assert len(component_params) > 0, "No parameters found in components to optimize"

    optimizer = optim.AdamW(component_params + gate_params, lr=config.lr, weight_decay=0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)
    logger.info(f"Base LR scheduler created: {config.lr_schedule}")

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
            components=components,
            causal_importances=causal_importances,
            causal_importances_upper_leaky=causal_importances_upper_leaky,
            target_out=target_out,
            device=device,
            n_params=n_params,
            faithfulness_scale_fn=faithfulness_scale_fn if config.lr_warmup_pct * config.steps > step and step > 1  else faithfulness_no_scale_fn,
        )

        log_data["loss/total"] = total_loss.item()
        log_data.update(loss_terms)

        with torch.inference_mode():
            # --- Logging --- #
            if step % config.print_freq == 0:
                tqdm.write(f"--- Step {step} ---")
                tqdm.write(f"LR: {step_lr:.6f}")
                tqdm.write(f"Total Loss: {log_data['loss/total']:.7f}")
                for name, value in loss_terms.items():
                    tqdm.write(f"{name}: {value:.7f}")

                if step > 0:
                    for layer_name, layer_alive_components in alive_components.items():
                        log_data[f"{layer_name}/n_alive_01"] = layer_alive_components.sum().item()
                        alive_components[layer_name] = torch.zeros(config.C, device=device).bool()

                # Calculate component logits and KL losses
                masked_component_logits = model.forward_with_components(
                    batch, components=components, masks=causal_importances
                )
                unmasked_component_logits = model.forward_with_components(
                    batch, components=components, masks=None
                )

                target_logits = model(batch)

                log_data["misc/unmasked_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=unmasked_component_logits, target=target_logits
                ).item()
                log_data["misc/masked_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=masked_component_logits, target=target_logits
                ).item()
                log_data["misc/unmasked_mse_vs_target"] = calc_mean_squared_error(
                    pred=unmasked_component_logits, target=target_logits
                ).item()
                log_data["misc/masked_mse_vs_target"] = calc_mean_squared_error(
                    pred=masked_component_logits, target=target_logits
                ).item()

                if config.log_ce_losses:
                    ce_losses = calc_ce_losses(
                        model=model,
                        batch=batch,
                        components=components,
                        masks=causal_importances,
                        unmasked_component_logits=unmasked_component_logits,
                        masked_component_logits=masked_component_logits,
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
                            unmasked_component_logits=unmasked_component_logits,
                            masked_component_logits=masked_component_logits,
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
                    if out_dir is not None:
                        for k, v in fig_dict.items():
                            v.savefig(out_dir / f"{k}_{step}.png")
                            tqdm.write(f"Saved plot to {out_dir / f'{k}_{step}.png'}")

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
            total_loss.backward(retain_graph=True)
            if config.faithfulness_scale == "rms" and rms_opt is not None:
                rms_opt.step()

            if step % config.print_freq == 0 and config.wandb_project:
                grad_norm: Float[Tensor, ""] = torch.zeros((), device=device)
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.flatten().pow(2).sum()  # type: ignore
                grad_norm_val = grad_norm.sqrt().item()
                wandb.log({"grad_norm": grad_norm_val}, step=step)

            optimizer.step()

    logger.info("Finished training loop.")
