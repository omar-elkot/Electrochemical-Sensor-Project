# Electrochemical Sensor Analysis Toolkit

A Python-based toolkit for simulating, analyzing, and visualizing electrochemical sensor measurements, with a focus on measurement uncertainty, environmental effects, sensor-to-sensor variability, and statistical model validation.

> **Note:** All measurements currently included in this project are simulated. The framework has not yet been validated against measurements from physical electrochemical sensors.

## Overview

Electrochemical sensors produce electrical signals that can depend on analyte concentration, environmental conditions, and differences between individual sensors. Understanding these measurements requires more than fitting a regression model: it also requires quantifying uncertainty, accounting for repeated measurements, examining model assumptions, and evaluating statistical reliability.

This project provides a modular computational framework for investigating these challenges. It combines synthetic data generation, data loading, regression analysis, diagnostic visualization, simulation-based validation, and an interactive dashboard.

**Central Research question:** How can statistical and computational methods be used to characterize electrochemical sensor measurements, quantify measurement uncertainty, and identify systematic effects caused by external environmental conditions, particularly, illumination?

## Key Features

- **Sensor simulation:** Generate synthetic measurements across analyte concentrations and illumination conditions, including measurement noise and sensor-specific variability.
- **Data processing:** Load and organize experiment data stored in CSV files, alongside experimental metadata.
- **Regression analysis:** Fit ordinary least squares (OLS) and mixed-effects models to investigate sensor response.
- **Uncertainty quantification:** Examine parameter estimates, confidence intervals, residual errors, and model diagnostics.
- **Statistical validation:** Investigate parameter recovery, false-positive behavior, model comparisons, and experimental design through repeated simulations.
- **Diagnostic visualization:** Generate response plots, residual plots, and Q–Q plots.
- **Interactive dashboard:** Explore sensor parameters, experimental conditions, and model behavior through Streamlit.

## Getting Started

There are two ways to use the project. Choose the one that suits you.

### Option 1 — The easier way, if you would like to see an overview: Use the interactive dashboard!

To explore the project without using Python or setting up a virtual environment? Use the interactive Streamlit dashboard:

**Open the online dashboard:** *Link coming soon.*

Once deployed, the dashboard will be publicly accessible through a browser link. You can share this link with anyone interested in exploring the project, without requiring them to install Python, download the repository, or configure any dependencies.

The dashboard provides an interactive overview of the project's functionality, making it easy to explore the sensor simulation and analysis without manually running individual Python scripts.

For those who want to inspect the source code, run their own simulations, or modify the analysis, the full setup instructions are provided in Option 2:

### Option 2 — The full workflow: Run the Python project locally

This option gives you direct access to the code, experiment files, statistical models, and validation scripts.

#### Requirements:

- Python 3.13 (tested)
- pip
- Git, if cloning the repository

#### 1. Get the project

Clone the repository:

```bash
git clone https://github.com/omar-elkot/electrochemical-sensor-project.git
cd electrochemical-sensor-project
```

Alternatively, download the repository from GitHub and open the project folder in your terminal.

#### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install the dependencies

From the project root, run:

```bash
python -m pip install -r requirements.txt
```

This installs the Python libraries needed to run the dashboard, analysis, and associated tests.

#### 4. Choose what to run

**A. Analyze the existing experiment**

The repository already contains a simulated experiment under `data/experiments/`. You do not need to generate another dataset to try the analysis.

Run:

```bash
python -m src.analyze_data
```

This executes the analysis workflow and generates diagnostic figures and terminal output. The workflow uses the experiment selected by the current project configuration.

**B. Generate a new simulated experiment**

To generate a new dataset, run:

```bash
python -m src.simulate
```

The simulation generates a new experiment containing sensor CSV files and metadata. Check the configured output location before running it, as reusing an existing destination may overwrite files in the repository.

After generating the experiment, analyze the latest dataset with:

```bash
python -m src.analyze_data --experiment latest
```
To analyze a specific experiment

If you have multiple experiments stored in data/experiments/, you can select a particular experiment by its ID. For example, to analyze experiment_001 run:

```bash
python -m src.analyze_data --experiment experiment_001
```
Replace experiment_001 with the ID of whichever experiment you want to analyze. The analysis workflow generates diagnostic figures and terminal output for the selected experiment.

**C. Open the dashboard locally: again, this is an easier alternative in which you do not need to simulate or execute the analysis workflow**

Instead of running the analysis scripts manually, launch the interactive dashboard:

```bash
streamlit run app.py
```

Streamlit will provide a local URL that you can open in your browser.

#### 5. Run the automated tests

To run the automated test suite:

```bash
python -m pytest
```

The tests check selected aspects of the statistical implementation and parameter-recovery behavior.

## Scientific Approach

The framework investigates how sensor output varies with analyte concentration and illumination. It also considers sensor-specific baseline offsets and the dependence between repeated measurements collected from the same sensor.

Ordinary least squares provides a baseline regression approach. Mixed-effects models can account for sensor-level variability by treating sensor identity as a grouping factor. Comparing these approaches helps investigate how modeling assumptions affect uncertainty estimates and statistical inference.

Simulation-based validation provides a way to test the analysis pipeline under controlled conditions where the underlying parameters are known. This makes it possible to investigate whether parameters can be recovered and whether statistical procedures behave as expected under the simulated assumptions.

## Project Structure

```text
electrochemical-sensor-project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── experiments/
│       └── experiment_001/
├── output/
│   └── experiments/
│       └── experiment_001/
│           └── figures/
├── results/
├── src/
│   ├── config.py
│   ├── simulate.py
│   ├── loader.py
│   ├── models.py
│   ├── plots.py
│   └── analyze_data.py
├── tests/
│   ├── model_diagnostics.py
│   ├── plot_diagnostics.py
│   └── test_parameter_recovery.py
└── validation_scripts/
```

- `app.py` — interactive Streamlit dashboard.
- `src/config.py` — project configuration and parameters.
- `src/simulate.py` — generation and storage of simulated experiments.
- `src/loader.py` — loading and processing experiment data.
- `src/models.py` — statistical model implementation.
- `src/plots.py` — visualization and diagnostic plots.
- `src/analyze_data.py` — analysis workflow.
- `tests/` — automated tests.
- `validation_scripts/` — simulation-based statistical validation workflows.
- `data/` — simulated experiment datasets.
- `output/` — generated figures and other analysis outputs.
- `results/` — saved validation and model-comparison outputs.

## Limitations and Future Work

The current framework uses simulated data rather than measurements collected from physical sensors. Its results therefore demonstrate computational and statistical methods, not the performance of a real electrochemical sensor.

Additional limitations and potential improvements include:

- **More realistic sensor behavior:** Investigate nonlinear responses, temporal drift, and additional sources of measurement noise.
- **Physical illumination measurements:** Lux is weighted to human visual sensitivity. Irradiance and wavelength would be more physically informative for studying illumination-dependent sensor effects.
- **Experimental validation:** Compare simulated behavior with measurements collected from physical sensors.
- **Model evaluation:** Extend the validation studies to investigate model assumptions, uncertainty estimates, and false-positive behavior under a wider range of conditions.

## Technologies

Python · NumPy · pandas · SciPy · statsmodels · Matplotlib · Plotly · Streamlit · pytest

## Project Status

The project includes simulation, statistical analysis, visualization, an interactive dashboard, and initial automated validation workflows. It is intended as a foundation for further development and eventual comparison with real sensor measurements.

## Author

`<Omar Elkot>`, `<B.Sc. Physics and Data Science>`, `<Constructor University, Bremen, Germany>`. `<https://www.linkedin.com/in/omar-mohamed-elkot/>`
