
"""Monte Carlo validation of OLS interaction-parameter recovery.

Uses simulated data with known ground-truth parameters to measure:
1) bias of the estimated interaction coefficient
2) RMSE of the estimates
3) empirical 95% confidence interval coverage

This validates OLS under the simulator's assumptions; it does not
establish performance on real electrochemical sensor measurements.
"""

import numpy as np
import pandas as pd

from src.config import SensorParams
from src.models import fit_ols
from src.simulate import simulate


def run_parameter_validation(
    n_simulations=200,
    n_sensors=8,
    n_reps=10,
    seed=2026,
):
    """Run repeated simulations and assess interaction recovery."""

    true_interaction = 0.0002
    rng = np.random.default_rng(seed)
    estimates = []

    params = SensorParams(
        baseline=100.0,
        b_conc=8.0,
        b_light=0.01,
        b_inter=true_interaction,
        noise_std=2.5,
        sensor_offset_std=3.0,
    )

    for run in range(n_simulations):
        data_seed = int(rng.integers(0, 2**31 - 1))

        df = simulate(
            params=params,
            n_sensors=n_sensors,
            n_reps=n_reps,
            seed=data_seed,
        )

        model = fit_ols(df)

        estimate = model.params[
            "concentration_centered:light_centered"
        ]
        ci = model.conf_int().loc[
            "concentration_centered:light_centered"
        ]

        estimates.append(
            {
                "simulation": run,
                "true_interaction": true_interaction,
                "estimated_interaction": float(estimate),
                "ci_lower": float(ci.iloc[0]),
                "ci_upper": float(ci.iloc[1]),
                "ci_width": float(ci.iloc[1] - ci.iloc[0]),
                "covered": bool(
                    ci.iloc[0] <= true_interaction <= ci.iloc[1]
                ),
            }
        )

    results = pd.DataFrame(estimates)
    errors = (
        results["estimated_interaction"] - true_interaction
    )

    summary = {
        "simulations": len(results),
        "true_interaction": true_interaction,
        "mean_estimate": results["estimated_interaction"].mean(),
        "bias": errors.mean(),
        "rmse": np.sqrt(np.mean(errors**2)),
        "ci_coverage": results["covered"].mean(),
        "mean_ci_width": results["ci_width"].mean(),
        "median_ci_width": results["ci_width"].median(),
    }

    return results, summary


if __name__ == "__main__":
    results, summary = run_parameter_validation()

    print("\nPARAMETER RECOVERY VALIDATION")
    print("-" * 40)

    for key, value in summary.items():
        if key == "ci_coverage":
            print(f"{key}: {value:.1%}")
        elif key in {"simulations"}:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.8g}")

    output_path = "output/validations/parameter_validation.csv"
    results.to_csv(output_path, index=False)
    print(f"\nIndividual results saved to {output_path}")
