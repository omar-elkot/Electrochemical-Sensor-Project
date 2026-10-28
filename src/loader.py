"""After making improvements in simulate.py, now that simulation is  seperated 
from the rest of the project, I created a dedicated data loader, since the data is now 
in formats of CSV files and a metadata json file, it's important to validate before analyzing.
What this will do is load the data from the eigt sensors (CSV files) and combine 
them into one data frame.

The improvement here is that instead of relying on 1 CSV file, the loader reads individual
files generated for each sensor and combined them into one dataset. It also checks the 
data structure and validates the measurements BEFORE they are analysed, helping preventing
errors caused by missing values or if measurements are invalid. """




from pathlib import Path
import json

import pandas as pd

from src.config import DATA_DIR

REQUIRED_COLUMNS = {
    "sensor_id",
    "concentration_mM",
    "light_lux",
    "replicate",
    "expected_mV",
    "sensor_offset_mV",
    "output_mV",
}


def list_experiments(root=DATA_DIR):
    """Return sorted IDs for saved experiment folders."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and (path / "metadata.json").is_file()
        and any(path.glob("sensor_*.csv"))
    )


def _resolve_experiment(experiment="latest", root=DATA_DIR):
    root = Path(root)
    if experiment == "latest":
        available = list_experiments(root)
        if not available:
            raise FileNotFoundError(f"No saved experiments found in {root}.")
        experiment = available[-1]

    candidate = Path(str(experiment))
    if candidate.is_absolute() or candidate.exists():
        folder = candidate
    else:
        folder = root / str(experiment)

    if not folder.is_dir():
        raise FileNotFoundError(f"Experiment not found: {folder}")
    return folder


def load_experiment(experiment="latest", root=DATA_DIR, verbose=True):
    """Load exactly one run by ID (e.g. experiment_002), 'latest', or path."""
    folder = _resolve_experiment(experiment, root)
    csv_files = sorted(folder.glob("sensor_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No sensor CSV files found in {folder}")

    frames = []
    for file in csv_files:
        part = pd.read_csv(file)
        missing = REQUIRED_COLUMNS - set(part.columns)
        if missing:
            raise ValueError(f"{file.name} is missing columns: {sorted(missing)}")
        if part.empty:
            raise ValueError(f"{file.name} contains no measurements.")
        try:
            file_sensor_id = int(file.stem.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Unexpected sensor filename: {file.name}") from exc
        if not (part["sensor_id"] == file_sensor_id).all():
            raise ValueError(
                f"Sensor IDs inside {file.name} do not match its filename."
            )
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError(f"The experiment in {folder} is empty.")
    if df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError(f"Missing values found in experiment {folder}.")
    if (df["replicate"] < 1).any():
        raise ValueError("Replicate numbers must be at least 1.")
    if (df["concentration_mM"] < 0).any():
        raise ValueError("Concentrations cannot be negative.")
    if (df["light_lux"] < 0).any():
        raise ValueError("Light levels cannot be negative.")
    if df.duplicated(
        ["sensor_id", "concentration_mM", "light_lux", "replicate"]
    ).any():
        raise ValueError("Duplicate sensor/condition/replicate rows found.")

    metadata_path = folder / "metadata.json"
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    expected_count = metadata.get("n_measurements")
    if expected_count is not None and int(expected_count) != len(df):
        raise ValueError(
            f"Metadata says {expected_count} measurements but CSVs contain {len(df)}."
        )

    if verbose:
        print(
            f"Loaded {len(df)} measurements from {len(csv_files)} sensor files "
            f"for {folder.name}."
        )
    df.attrs["experiment_id"] = folder.name
    df.attrs["experiment_folder"] = str(folder)
    return df


if __name__ == "__main__":
    experiments = list_experiments()
    print("Available experiments:", experiments)
    if experiments:
        df = load_experiment(experiments[-1])
        print(df.head())
        print(f"Unique sensors: {df['sensor_id'].nunique()}")
