
"""Run the electrochemical sensor scientific validation suite.
Experiments:
1. Parameter recovery
2. OLS vs mixed-effects model comparison
3. False-positive rate under the null hypothesis
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SensorParams
from src.models import fit_ols, fit_mixed
from src.simulate import simulate
from validation_scripts.parameter_validation import run_parameter_validation
from validation_scripts.false_positive_validation import run_false_positive_validation


TRUE_INTERACTION = 0.0002
INTERACTION_TERM = "concentration_centered:light_centered"
OUTPUT_DIR = Path("output/validations")


def evaluate_model(model, model_name, true_value, simulation):
    """Extract the interaction estimate and its confidence interval."""
    ci = model.conf_int().loc[INTERACTION_TERM]
    estimate = float(model.params[INTERACTION_TERM])
    lower = float(ci.iloc[0])
    upper = float(ci.iloc[1])

    return {
        "simulation": simulation,
        "model": model_name,
        "estimate": estimate,
        "bias": estimate - true_value,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_width": upper - lower,
        "covered": lower <= true_value <= upper,
    }


def run_comparison(n_simulations=100, seed=2026):
    """Compare OLS and MixedLM on identical simulated datasets."""
    rng = np.random.default_rng(seed)

    params = SensorParams(
        baseline=100.0,
        b_conc=8.0,
        b_light=0.01,
        b_inter=TRUE_INTERACTION,
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

        for model_name, fit_function in (
            ("OLS", fit_ols),
            ("MixedLM", fit_mixed),
        ):
            try:
                if model_name == "MixedLM":
                    model = fit_function(df, verbose=False)
                else:
                    model = fit_function(df)

                records.append(
                    evaluate_model(
                        model,
                        model_name,
                        TRUE_INTERACTION,
                        simulation,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "simulation": simulation,
                        "model": model_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if (simulation + 1) % 10 == 0:
            print(
                f"Model comparison: {simulation + 1}/{n_simulations}"
            )

    results = pd.DataFrame(records)
    failure_df = pd.DataFrame(
        failures,
        columns=["simulation", "model", "error"],
    )

    if results.empty:
        raise RuntimeError("Neither model produced usable results.")

    summaries = []

    for model_name in ("OLS", "MixedLM"):
        group = results[results["model"] == model_name]

        if group.empty:
            summaries.append({
                "model": model_name,
                "successful_fits": 0,
                "failed_fits": n_simulations,
                "bias": np.nan,
                "rmse": np.nan,
                "ci_coverage": np.nan,
                "mean_ci_width": np.nan,
            })
            continue

        errors = group["estimate"] - TRUE_INTERACTION

        summaries.append({
            "model": model_name,
            "successful_fits": len(group),
            "failed_fits": n_simulations - len(group),
            "bias": errors.mean(),
            "rmse": np.sqrt(np.mean(errors**2)),
            "ci_coverage": group["covered"].mean(),
            "mean_ci_width": group["ci_width"].mean(),
        })

    return results, pd.DataFrame(summaries), failure_df


def main():
    """Run all validation experiments and save their outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ELECTROCHEMICAL SENSOR SCIENTIFIC VALIDATION")
    print("=" * 60)

    # 1. Parameter recovery
    print("\n[1/3] PARAMETER RECOVERY")
    recovery_results, recovery_summary = run_parameter_validation(
        n_simulations=200,
        seed=2026,
    )
    recovery_results.to_csv(
        OUTPUT_DIR / "parameter_validation.csv",
        index=False,
    )

    for key, value in recovery_summary.items():
        print(f"  {key}: {value}")

    # 2. OLS vs MixedLM
    print("\n[2/3] MODEL COMPARISON")
    comparison_results, comparison_summary, comparison_failures = (
        run_comparison(n_simulations=100, seed=2026)
    )

    comparison_results.to_csv(
        OUTPUT_DIR / "model_comparison_validation.csv",
        index=False,
    )
    comparison_failures.to_csv(
        OUTPUT_DIR / "model_comparison_failures.csv",
        index=False,
    )
    print(comparison_summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.6g}",
    ))

    # 3. False-positive validation
    print("\n[3/3] FALSE-POSITIVE VALIDATION")
    fp_results, fp_summary, fp_failures = (
        run_false_positive_validation(
            n_simulations=100,
            seed=2027,
        )
    )

    fp_results.to_csv(
        OUTPUT_DIR / "false_positive_validation.csv",
        index=False,
    )
    fp_failures.to_csv(
        OUTPUT_DIR / "false_positive_failures.csv",
        index=False,
    )
    print(fp_summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    ))

    print("\n" + "=" * 60)
    print("VALIDATION RUN COMPLETE")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
