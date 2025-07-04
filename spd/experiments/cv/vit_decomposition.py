from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import fire
import matplotlib.pyplot as plt
import yaml
from jaxtyping import Float

import wandb
from torch import Tensor

from spd.configs import Config, CVTaskConfig  # type: ignore
from spd.data import create_image_data_loader, VisionDatasetConfig
from spd.log import logger
from spd.plotting import plot_mean_component_activation_counts
from spd.run_spd import get_common_run_name_suffix, optimize
from spd.utils import get_device, load_config, load_pretrained, set_seed
from spd.wandb_utils import init_wandb

# ---------------------------------------------------------------------------
# Helper: derive run‑name identical to LM script but w/ vision specifics
# ---------------------------------------------------------------------------


def get_run_name(
    config: Config,
    pretrained_model_name: str | None,
) -> str:
    """Generate a run name based on the config."""
    assert isinstance(config.task_config, CVTaskConfig)
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        if pretrained_model_name:
            run_suffix += f"_pretrained{pretrained_model_name}"
    return config.wandb_run_name_prefix + run_suffix

def plot_vision_results(
    mean_component_activation_counts: dict[str, Float[Tensor, " m"],],
) -> plt.Figure:
    return plot_mean_component_activation_counts(mean_component_activation_counts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config_path_or_obj: Path | str | Config) -> None:
    # Load & (optionally) init WandB
    config = load_config(config_path_or_obj, config_model=Config)
    if config.wandb_project:
        config = init_wandb(config, config.wandb_project)

    # Safety: assert vision task
    assert isinstance(config.task_config, CVTaskConfig), (
        "For this script task_config must be CVTaskConfig",
    )

    set_seed(config.seed)
    logger.info(config)

    device = get_device()
    logger.info(f"Using device: {device}")

    # ---------------- Model ----------------
    logger.info("Loading pre‑trained vision model …")
    target_model = load_pretrained(
        path_to_class=config.pretrained_model_class,
        model_path=config.pretrained_model_path,
        model_name_hf=config.pretrained_model_name_hf,
    )

    # -------------- Run‑name + outdir --------------
    run_name = get_run_name(config, pretrained_model_name=config.pretrained_model_name_hf)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialised"
        wandb.run.name = run_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # -------- Save config --------
    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    # ---------------- Data ----------------
    logger.info("Preparing vision dataset …")
    train_ds_cfg = VisionDatasetConfig(
        name=config.task_config.dataset_name, # type: ignore
        split=config.task_config.train_data_split,  # type: ignore
        hf_image_processor_path=config.pretrained_model_name_hf
    )

    train_loader, _ = create_image_data_loader(  # type: ignore[arg-type]
        ds_cfg=train_ds_cfg,  # VisionDatasetConfig is compatible
        batch_size=config.batch_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
        trust_remote_code=True,  # for ViT models
        shuffle=True,
        balanced=False,
    )

    eval_ds_cfg = VisionDatasetConfig(
        name=config.task_config.dataset_name,  # type: ignore
        split=config.task_config.eval_data_split,  # type: ignore
        hf_image_processor_path=config.pretrained_model_name_hf,
    )

    eval_loader, _ = create_image_data_loader(  # type: ignore[arg-type]
        ds_cfg=eval_ds_cfg,
        batch_size=config.batch_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
        trust_remote_code=True,  # for ViT models
        shuffle=False,  # eval should not shuffle
        balanced=False
    )

    logger.info("Dataset ready – starting optimisation …")

    assert config.n_eval_steps is not None, "n_eval_steps must be set"
    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
    )

    logger.info("Optimisation complete.")

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
