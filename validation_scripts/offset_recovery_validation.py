"""Validate sensor-offset recovery as repetitions per condition increase.
The same simulated sensor offsets are reused at 2, 5, and 10 repetitions,
so the comparison is paired within each simulation run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SensorParams, N_SENSORS
from src.simulate import simulate
from src.models import fit_mixed

REPETITION_COUNTS = (2, 5, 10)


def run_validation(n_runs=100, seed=1000, output_dir="output/validations"):
    if n_runs < 2:
        raise ValueError("Use at least two simulation runs for uncertainty estimates.")

    params = SensorParams()
    records = []
    failures = []

    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)
        offsets = rng.normal(0, params.sensor_offset_std, N_SENSORS)

        for n_reps in REPETITION_COUNTS:
            df = simulate(
                params=params,
                n_sensors=N_SENSORS,
                n_reps=n_reps,
                seed=seed + run,
                sensor_offsets=offsets,
            )
            record = {
                "run": run,
                "n_reps": n_reps,
                "converged": False,
                "usable_random_effects": False,
                "offset_rmse_mV": np.nan,
                "estimated_sensor_variance": np.nan,
            }
            try:
                model = fit_mixed(df, verbose=False)
                record["converged"] = bool(model.converged)
                variance = float(model.cov_re.iloc[0, 0])
                record["estimated_sensor_variance"] = variance

                if not model.converged:
                    failures.append((run + 1, n_reps, "not converged"))
                elif not np.isfinite(variance) or variance <= 1e-8:
                    failures.append((run + 1, n_reps, "boundary/singular random variance"))
                else:
                    true_offsets = (
                        df.groupby("sensor_id")["sensor_offset_mV"]
                        .first().sort_index().astype(float)
                    )
                    estimated_offsets = pd.Series({
                        sensor_id: float(effect.iloc[0])
                        for sensor_id, effect in model.random_effects.items()
                    }).sort_index()

                    common = true_offsets.index.intersection(estimated_offsets.index)
                    if len(common) != N_SENSORS:
                        failures.append((run + 1, n_reps, "sensor random effects incomplete"))
                    else:
                        # Random effects represent deviations from the population
                        # intercept, so compare centered true and estimated offsets.
                        truth = true_offsets.loc[common]
                        estimate = estimated_offsets.loc[common]
                        truth = truth - truth.mean()
                        estimate = estimate - estimate.mean()
                        record["offset_rmse_mV"] = float(
                            np.sqrt(np.mean((estimate - truth) ** 2))
                        )
                        record["usable_random_effects"] = True
            except Exception as exc:
                failures.append((run + 1, n_reps, f"{type(exc).__name__}: {exc}"))

            records.append(record)

    results = pd.DataFrame(records)
    valid = results[results["usable_random_effects"]].copy()
    print("\n========== SENSOR-OFFSET RECOVERY VALIDATION ==========")
    print(f"Requested fits: {len(results)} ({n_runs} runs × {len(REPETITION_COUNTS)} repetition settings)")
    print(f"Usable offset-recovery fits: {len(valid)} / {len(results)} ({100*len(valid)/len(results):.1f}%)")
    print(f"Failed or unusable fits: {len(failures)}")
    if failures:
        print("First 15 failures:")
        for failure in failures[:15]:
            print(f"  Run {failure[0]}, repetitions={failure[1]}: {failure[2]}")

    print("\nOffset RMSE by repetition count (usable fits only):")
    for n_reps in REPETITION_COUNTS:
        values = valid.loc[valid["n_reps"] == n_reps, "offset_rmse_mV"].dropna()
        if len(values):
            print(
                f"{n_reps:2d} reps: n={len(values):3d}, "
                f"mean RMSE={values.mean():.3f} mV, SD={values.std(ddof=1):.3f} mV"
            )
        else:
            print(f"{n_reps:2d} reps: no usable fits")

    wide = valid.pivot(index="run", columns="n_reps", values="offset_rmse_mV")
    print("\nPaired RMSE differences (first minus second; approximate 95% CI):")
    for first, second in [(2, 5), (5, 10), (2, 10)]:
        if first not in wide or second not in wide:
            continue
        differences = (wide[first] - wide[second]).dropna()
        if len(differences) < 2:
            print(f"{first} - {second} reps: insufficient paired fits")
            continue
        mean = differences.mean()
        se = differences.std(ddof=1) / np.sqrt(len(differences))
        print(
            f"{first} - {second} reps (n={len(differences)}): "
            f"{mean:.3f} mV [{mean - 1.96*se:.3f}, {mean + 1.96*se:.3f}]"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "offset_recovery_validation.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nDetailed validation results saved to: {csv_path}")
    return results


if __name__ == "__main__":
    run_validation()
