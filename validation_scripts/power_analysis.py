"""Mixed-effects power and false-positive checks for the interaction term.

Run from the repository root:
    python -m scripts.power_analysis

This simulation includes sensor-specific random intercepts and tests the
interaction with the mixed model, rather than treating all measurements as
independent.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SensorParams, N_SENSORS, N_REPS, RANDOM_SEED
from src.models import fit_mixed
from src.simulate import simulate


def run_power_analysis(n_simulations=200, seed=RANDOM_SEED, output_dir="output/validations"):
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")

    base_params = SensorParams()
    scenarios = {
        "Null (b_inter = 0)": replace(base_params, b_inter=0.0),
        "Interaction present": base_params,
    }
    rows = []

    for scenario_index, (scenario, params) in enumerate(scenarios.items()):
        p_values = []
        failures = 0
        for run in range(n_simulations):
            # Distinct but reproducible data in each replicate.
            run_seed = seed + scenario_index * 100_000 + run
            df = simulate(
                params=params,
                n_sensors=N_SENSORS,
                n_reps=N_REPS,
                seed=run_seed,
            )
            try:
                model = fit_mixed(df, verbose=False)
                if not model.converged:
                    failures += 1
                    continue
                random_variance = float(model.cov_re.iloc[0, 0])
                if not np.isfinite(random_variance) or random_variance <= 1e-8:
                    failures += 1
                    continue
                p_value = float(
                    model.pvalues.get("concentration_centered:light_centered", np.nan)
                )
                if np.isfinite(p_value):
                    p_values.append(p_value)
                else:
                    failures += 1
            except Exception:
                failures += 1

        rate = 100 * np.mean(np.asarray(p_values) < 0.05) if p_values else np.nan
        rows.append({
            "scenario": scenario,
            "requested_simulations": n_simulations,
            "usable_fits": len(p_values),
            "failed_or_unusable_fits": failures,
            "detection_or_false_positive_rate_pct": rate,
        })

    results = pd.DataFrame(rows)
    print("\n========== MIXED-MODEL INTERACTION CHECK ==========")
    print(results.to_string(index=False, formatters={
        "detection_or_false_positive_rate_pct": lambda x: f"{x:.1f}%" if pd.notna(x) else "n/a"
    }))
    print(
        "\nInterpretation: the null scenario estimates the false-positive rate; "
        "the interaction-present scenario estimates detection power. With a "
        "finite number of simulations, both rates have Monte Carlo uncertainty."
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "interaction_power_analysis.csv"
    results.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path}")
    return results


if __name__ == "__main__":
    run_power_analysis()
