
""" Improvement: After cleaning up the project structure and centralizing the parameters in config.py, 
I decided to improve the simulation itself by separating the data generation process from the rest of 
the code. (I added the following changes from the old simulate.py):
I created a simulate() function that can be called by other modules without automatically running
the entire experiment whenever the file is imported. I also introduced sensor-specific offsets
to account for differences between individual sensors, while keeping these separate from the random 
measurement noise. Additionally, I made it possible to simulate both linear and saturating illumination 
responses, allowing the model to represent more complex behavior and later investigate whether a simpler 
linear model is sufficient. Finally, I added a function to export the generated measurements into separate
CSV files for each sensor, together with a metadata file containing the experimental configuration. These
changes make the simulator more flexible, reusable, and suitable for testing different experimental conditions
and analysis methods. Now that this works, I created loader.py to validate the CSV files and analyze.py 
to implement and test and models.py for the statistical models"""

from itertools import product
from pathlib import Path
import json
import re
import shutil

import numpy as np
import pandas as pd

from src.config import (
    SensorParams,
    CONCENTRATIONS,
    LIGHT_LEVELS,
    N_SENSORS,
    N_REPS,
    RANDOM_SEED,
    DATA_DIR,
)


def simulate(
    params=None,
    concentrations=None,
    light_levels=None,
    n_sensors=N_SENSORS,
    n_reps=N_REPS,
    seed=RANDOM_SEED,
    sensor_offsets=None,
):
    """Generate simulated measurements and return them as a DataFrame."""
    params = params or SensorParams()
    concentrations = list(CONCENTRATIONS if concentrations is None else concentrations)
    light_levels = list(LIGHT_LEVELS if light_levels is None else light_levels)

    if not isinstance(n_sensors, int) or n_sensors < 2:
        raise ValueError("n_sensors must be an integer of at least 2.")
    if not isinstance(n_reps, int) or n_reps < 1:
        raise ValueError("n_reps must be a positive integer.")
    if not concentrations or not light_levels:
        raise ValueError("Concentration and light-level lists cannot be empty.")
    if not np.isfinite(concentrations).all() or any(value < 0 for value in concentrations):
        raise ValueError("Concentrations must be finite and non-negative.")
    if not np.isfinite(light_levels).all() or any(value < 0 for value in light_levels):
        raise ValueError("Light levels must be finite and non-negative.")
    if params.noise_std < 0 or params.sensor_offset_std < 0:
        raise ValueError("Noise and sensor-offset standard deviations cannot be negative.")
    if params.light_scale is not None and params.light_scale <= 0:
        raise ValueError("light_scale must be positive when supplied.")

    rng = np.random.default_rng(seed)

    if sensor_offsets is None:
        offsets = rng.normal(0, params.sensor_offset_std, n_sensors)
    else:
        offsets = np.asarray(sensor_offsets, dtype=float)
        if offsets.shape != (n_sensors,) or not np.isfinite(offsets).all():
            raise ValueError("sensor_offsets must contain one finite value per sensor.")

    rows = []
    for sensor_id, concentration, light, replicate in product(
        range(n_sensors),
        concentrations,
        light_levels,
        range(1, n_reps + 1),
    ):
        effective_light = light
        if params.light_scale is not None:
            effective_light = params.light_scale * (
                1 - np.exp(-light / params.light_scale)
            )

        underlying_response = (
            params.baseline
            + params.b_conc * concentration
            + params.b_light * effective_light
            + params.b_inter * concentration * effective_light
        )
        expected_sensor_output = underlying_response + offsets[sensor_id]
        measured_output = expected_sensor_output + rng.normal(0, params.noise_std)

        rows.append({
            "sensor_id": sensor_id,
            "concentration_mM": concentration,
            "light_lux": light,
            "replicate": replicate,
            # Underlying response excludes the sensor-specific offset.
            "expected_mV": underlying_response,
            "sensor_offset_mV": offsets[sensor_id],
            # Expected output for this particular sensor includes its offset.
            "expected_sensor_output_mV": expected_sensor_output,
            "output_mV": measured_output,
        })

    return pd.DataFrame(rows)


def _next_experiment_id(root):
    """Return the next unused experiment_NNN identifier."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^experiment_(\d+)$")
    numbers = [
        int(match.group(1))
        for path in root.iterdir()
        if path.is_dir()
        for match in [pattern.match(path.name)]
        if match
    ]
    number = max(numbers, default=0) + 1
    while (root / f"experiment_{number:03d}").exists():
        number += 1
    return f"experiment_{number:03d}"


def write_experiment(
    df,
    folder=DATA_DIR,
    params=None,
    seed=RANDOM_SEED,
    experiment_id=None,
):
    """Save a run to a new folder and return that folder's path.

    `folder` is the parent directory for experiments, not the run folder.
    """
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    experiment_id = experiment_id or _next_experiment_id(root)

    if not re.fullmatch(r"experiment_[A-Za-z0-9_-]+", experiment_id):
        raise ValueError("experiment_id must look like 'experiment_001'.")

    destination = root / experiment_id
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists. Choose another ID; existing runs are never overwritten."
        )
    if df.empty:
        raise ValueError("Cannot save an empty experiment.")

    required = {
        "sensor_id", "concentration_mM", "light_lux", "replicate",
        "expected_mV", "sensor_offset_mV", "output_mV",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot save dataset; missing columns: {sorted(missing)}")

    destination.mkdir()
    try:
        for sensor_id, sensor_df in df.groupby("sensor_id"):
            sensor_df.to_csv(
                destination / f"sensor_{int(sensor_id):03d}.csv",
                index=False,
            )

        params = params or SensorParams()
        metadata = {
            "experiment_id": experiment_id,
            "seed": int(seed) if seed is not None else None,
            "n_sensors": int(df["sensor_id"].nunique()),
            "n_measurements": int(len(df)),
            "model_parameters": {
                key: getattr(params, key)
                for key in params.__dataclass_fields__
            },
            "concentrations_mM": sorted(df["concentration_mM"].unique().tolist()),
            "light_levels_lux": sorted(df["light_lux"].unique().tolist()),
            "sensor_offsets_mV": {
                str(int(sensor_id)): float(sensor_df["sensor_offset_mV"].iloc[0])
                for sensor_id, sensor_df in df.groupby("sensor_id")
            },
            "note": (
                "Synthetic data, not physical sensor measurements. Lux is a "
                "simplified illumination proxy; wavelength and irradiance are not modeled."
            ),
        }
        with (destination / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
    except Exception:
        # Clean up only the new incomplete folder, never older experiments.
        shutil.rmtree(destination, ignore_errors=True)
        raise

    print(f"Saved {len(df)} measurements to {destination}")
    return destination


if __name__ == "__main__":
    params = SensorParams()
    df = simulate(params=params)
    saved_folder = write_experiment(df, params=params)
    print(df.head())
    print(f"Total measurements: {len(df)}")
    print(f"Experiment ID: {saved_folder.name}")

