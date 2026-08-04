from __future__ import annotations

from importlib import import_module
from pathlib import Path


def _import_plot_main(module_name: str):
    """Import main() from a plotting module."""
    module = import_module(module_name)

    if not hasattr(module, "main"):
        raise AttributeError(f"{module_name} does not define main().")

    return module.main


def run_auto_plots(args) -> None:
    """Automatically run metric plotting scripts after training/evaluation.

    JSON input:
        project_root / experiments / utd_mhad_rgb_inertial / logs

    PNG output:
        project_root / experiments / utd_mhad_rgb_inertial / logs / overview_plots
    """
    if not getattr(args, "auto_plot", False):
        return

    exp_dir = Path(getattr(args, "exp_dir", "experiments/utd_mhad_rgb_inertial"))
    log_dir = exp_dir / "logs"
    save_dir = log_dir / "overview_plots"

    print("=" * 80)
    print("[Auto Plot] Start plotting training curves.")
    print(f"[Auto Plot] JSON LOG_DIR = {log_dir}")
    print(f"[Auto Plot] PNG SAVE_DIR = {save_dir}")
    print("=" * 80)

    try:
        plot_metrics_main = _import_plot_main("plot.plot_metrics")
        plot_overview_main = _import_plot_main("plot.plot_training_overview")

        plot_metrics_main(
            log_dir=log_dir,
            save_dir=save_dir,
        )

        plot_overview_main(
            log_dir=log_dir,
            save_dir=save_dir,
        )

    except Exception as error:
        print("[Auto Plot] Plotting failed, but training/evaluation has finished.")
        print(f"[Auto Plot] Error: {error}")

    print("=" * 80)
    print("[Auto Plot] Finished.")
    print("=" * 80)