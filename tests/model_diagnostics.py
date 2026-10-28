
from src.loader import load_experiment
from src.models import (
    fit_ols,
    fit_mixed,
    coefficient_confidence_intervals,
    compare_ols_models,
)
import numpy as np
import pandas as pd

df = load_experiment()

print("\n========== OLS REGRESSION ==========")
ols = fit_ols(df)
print(ols.summary())

print("\n========== OLS CONFIDENCE INTERVALS ==========")
print(coefficient_confidence_intervals(ols))

print("\n========== MODEL COMPARISON ==========")
comparison = compare_ols_models(df)
print(f"F-statistic: {comparison['f_statistic']:.4f}")
print(f"P-value: {comparison['p_value']:.6f}")

print("\n========== MIXED-EFFECTS REGRESSION ==========")
mixed = fit_mixed(df)
print(mixed.summary())

print("\n========== MIXED-MODEL CONFIDENCE INTERVALS ==========")
print(coefficient_confidence_intervals(mixed))



print("\n========== QUANTITATIVE ERROR ANALYSIS ==========")

models = {
    "OLS": ols,
    "Mixed-effects": mixed,
}

for name, model in models.items():
    residuals = np.asarray(model.resid)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f} mV")
    print(f"MAE:  {mae:.4f} mV")
    print(f"Residual mean: {np.mean(residuals):.4f} mV")
    print(f"Residual SD: {np.std(residuals, ddof=1):.4f} mV")

print("\n========== RESIDUALS BY SENSOR (MIXED MODEL) ==========")

df_results = df[["sensor_id"]].copy()
df_results["residual_mV"] = np.asarray(mixed.resid)

sensor_summary = (
    df_results.groupby("sensor_id")["residual_mV"]
    .agg(["mean", "std", "count"])
    .rename(columns={
        "mean": "mean_residual_mV",
        "std": "residual_sd_mV",
        "count": "n_measurements",
    })
)

print(sensor_summary.round(4))



print("\n========== SENSOR OFFSET RECOVERY ==========")

# The simulator stores the true offset for each sensor.
true_offsets = (
    df.groupby("sensor_id")["sensor_offset_mV"]
    .first()
    .sort_index()
)

# Extract the sensor-specific random intercept estimates.
estimated_offsets = pd.Series(
    {
        sensor_id: effects.iloc[0]
        for sensor_id, effects in mixed.random_effects.items()
    },
    name="estimated_offset_mV",
).sort_index()

offset_comparison = pd.concat(
    [
        true_offsets.rename("true_offset_mV"),
        estimated_offsets,
    ],
    axis=1,
)

offset_comparison["error_mV"] = (
    offset_comparison["estimated_offset_mV"]
    - offset_comparison["true_offset_mV"]
)

print(offset_comparison.round(3))

offset_rmse = np.sqrt(np.mean(offset_comparison["error_mV"] ** 2))

print(f"\nOffset recovery RMSE: {offset_rmse:.4f} mV")
print(
    "Offset correlation:",
    f"{offset_comparison['true_offset_mV'].corr(offset_comparison['estimated_offset_mV']):.4f}",
)
