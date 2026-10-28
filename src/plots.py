
"""
Finally, here are some visualisation tools for the electrochemical sensor project.

This module generates:
1.Sensor response curves.
2.Residuals-versus-fitted diagnostic plots.
3.Residual Q-Q plots.
4.Mixed-effects model residuals-versus-fitted diagnostic plot.
5.Mixed-effects model residual Q-Q plot.

All figures are saved to the output folder rather than displayed
only on screen, so they can later be included in the project report.
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

FIGURE_DIR = Path("output/figures")


def plot_sensor_response(df, output_dir=FIGURE_DIR):
    """Plot condition means with uncertainty across sensors.
    Measurements are first averaged within each sensor and condition; error
    bars then show the standard error across sensor means, avoiding treating
    repeated measurements from one sensor as independent sensors """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_sensor = (
        df.groupby(["sensor_id", "concentration_mM", "light_lux"])["output_mV"]
        .mean()
        .reset_index(name="sensor_mean_mV")
    )
    summary = (
        by_sensor.groupby(["concentration_mM", "light_lux"])["sensor_mean_mV"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])

    fig, ax = plt.subplots(figsize=(9, 6))
    for light_level, group in summary.groupby("light_lux"):
        group = group.sort_values("concentration_mM")
        ax.errorbar(
            group["concentration_mM"], group["mean"], yerr=group["sem"],
            marker="o", capsize=4, label=f"{light_level:g} lux",
        )
    ax.set(
        xlabel="Concentration (mM)",
        ylabel="Mean sensor output (mV)",
        title="Electrochemical Sensor Response",
    )
    ax.legend(title="Illumination")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "sensor_response.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved sensor response plot: {path}")


def plot_residuals(model, output_dir=FIGURE_DIR, model_name="ols"):
    """Save residuals-versus-fitted diagnostic plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted = np.asarray(model.fittedvalues)
    residuals = np.asarray(model.resid)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fitted, residuals, alpha=0.55)
    ax.axhline(0, linestyle="--")
    ax.set(
        xlabel="Fitted output (mV)", ylabel="Residual (mV)",
        title=f"Residuals vs. Fitted Values ({model_name})",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / f"{model_name}_residuals.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved residual plot: {path}")


def plot_residual_qq(model, output_dir=FIGURE_DIR, model_name="ols"):
    """Save a residual Q-Q plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = sm.qqplot(np.asarray(model.resid), line="45", fit=True)
    fig.set_size_inches(9, 6)
    fig.suptitle(f"Residual Q-Q Plot ({model_name})")
    fig.tight_layout()
    path = output_dir / f"{model_name}_qq.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved Q-Q plot: {path}")
