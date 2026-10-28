
"""Check false-positive rates when the true interaction is zero."""

import numpy as np
import pandas as pd

from src.config import SensorParams
from src.models import fit_ols, fit_mixed
from src.simulate import simulate

ALPHA = 0.05
INTERACTION_TERM = "concentration_centered:light_centered"


def run_false_positive_validation(n_simulations=100, seed=2027):
    rng = np.random.default_rng(seed)

    params = SensorParams(
        baseline=100.0,
        b_conc=8.0,
        b_light=0.01,
        b_inter=0.0,  # Null hypothesis: no interaction exists
        noise_std=2.5,
        sensor_offset_std=3.0,
    )

    records = []
    failures = []

    for simulation in range(n_simulations):
        data_seed = int(rng.integers(0, 2**31 - 1))

        df = simulate(
            params=params,
            n_sensors=8,
            n_reps=10,
            seed=data_seed,
        )

        for name, fit_function in (
            ("OLS", fit_ols),
            ("MixedLM", fit_mixed),
        ):
            try:
                model = (
                    fit_function(df, verbose=False)
                    if name == "MixedLM"
                    else fit_function(df)
                )

                p_value = float(model.pvalues[INTERACTION_TERM])

                records.append(
                    {
                        "simulation": simulation,
                        "model": name,
                        "p_value": p_value,
                        "false_positive": p_value < ALPHA,
                    }
                )

            except Exception as exc:
                failures.append(
                    {
                        "simulation": simulation,
                        "model": name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if (simulation + 1) % 10 == 0:
            print(
                f"Completed {simulation + 1}/{n_simulations} simulations"
            )

    results = pd.DataFrame(records)
    failure_df = pd.DataFrame(
        failures, columns=["simulation", "model", "error"]
    )

    summaries = []

    for name in ("OLS", "MixedLM"):
        group = results[results["model"] == name]
        n_success = len(group)

        summaries.append(
            {
                "model": name,
                "successful_fits": n_success,
                "failed_fits": n_simulations - n_success,
                "false_positives": int(group["false_positive"].sum()),
                "false_positive_rate": (
                    group["false_positive"].mean()
                    if n_success else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summaries)

    return results, summary, failure_df


if __name__ == "__main__":
    results, summary, failures = run_false_positive_validation(
        n_simulations=100
    )

    print("\nFALSE-POSITIVE VALIDATION")
    print("True interaction coefficient: 0.0")
    print(f"Significance threshold: {ALPHA}")
    print("\nSummary:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    results.to_csv("output/validations/false_positive_validation.csv", index=False)
    failures.to_csv("output/validations/false_positive_failures.csv", index=False)

    print("\nSaved:")
    print("  output/validations/false_positive_validation.csv")
    print("  output/validations/false_positive_failures.csv")

