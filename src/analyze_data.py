
""" This file runs the complete analysis of the simulated electrochemical sensor experiment.
The workflow is:
1.Generate measurements using the simulator: python -m src.simulate
Disclaimer: This file won't work unless pre-saved files are available from simulate.py. For
your convenience, there's pre-saved data in the repository, so you do not need to generate more.
This code then:
2.Loads and validates the individual sensor CSV files (using loader.py)
3.Summarizes the measurements for each experimental condition.
4.Fits statistical models to estimate the effects of concentration
and illumination on the sensor output (using models.py)
5. Quantify uncertainty and compare models with and without an interaction term.
6. Generate and save plots for the report.

The analysis uses both ordinary least squares (OLS) regression and
mixed-effects regression. The mixed-effects model accounts for persistent
differences between individual sensors.

How to use this file: depends if you generated your own simulated data, or used the pre-existing one)
   Examples:
    python -m src.analyze_data
    python -m src.analyze_data --experiment experiment_001
    python -m src.analyze_data --experiment latest
"""


from pathlib import Path
import argparse
import numpy as np

from src.loader import load_experiment, list_experiments
from src.models import (
    coefficient_confidence_intervals,
    compare_ols_models,
    fit_mixed,
    fit_ols,
)
from src.plots import (
    plot_residual_qq,
    plot_residuals,
    plot_sensor_response,
)


def main(experiment="latest"):
    # Check that at least one saved experiment is available.
    available = list_experiments()

    if not available:
        raise FileNotFoundError(
            "No saved experiments found. Run `python -m src.simulate` first."
        )

    # Load the requested experiment. Selecting an explicit experiment ID
    # prevents the analysis from accidentally using a different dataset.
    df = load_experiment(experiment)

    experiment_id = df.attrs["experiment_id"]
    experiment_folder = df.attrs["experiment_folder"]

    # Save each experiment's results separately. This prevents a new
    # analysis from overwriting figures generated for an earlier run.
    output_dir = Path("output") / "experiments" / experiment_id
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalyzing {experiment_id}")
    print(f"Saved data folder: {experiment_folder}")
    print(f"Results folder:    {output_dir}")
    print(f"Total measurements: {len(df)}")
    print(f"Number of sensors: {df['sensor_id'].nunique()}")

    # Display the experimental conditions represented in the dataset.
    print(
        "Concentration levels (mM):",
        sorted(df["concentration_mM"].unique()),
    )
    print(
        "Illumination levels (lux):",
        sorted(df["light_lux"].unique()),
    )

    # Summarize the distributions of the experimental variables and
    # measured sensor output. These statistics provide an initial
    # overview before fitting any statistical models.
    print("\n========== DESCRIPTIVE STATISTICS ==========")
    print(
        df[["concentration_mM", "light_lux", "output_mV"]]
        .describe()
        .round(4)
    )

    # Group measurements by concentration and illumination.
    # The mean describes the average sensor response, the standard
    # deviation describes the spread, and count reports the number
    # of measurements available for each condition.
    grouped = (
        df.groupby(["concentration_mM", "light_lux"])["output_mV"]
        .agg(["mean", "std", "count"])
    )

    print("\n========== CONDITION SUMMARY ==========")
    print(grouped.round(3).to_string())

    # Fit the ordinary least-squares (OLS) regression model.
    # This model estimates how concentration and illumination relate
    # to sensor output, including a possible interaction between them.
    ols = fit_ols(df)

    # Compare the full OLS model with a reduced model that excludes
    # the concentration-illumination interaction term.
    comparison = compare_ols_models(df)

    print("\n========== OLS REGRESSION MODEL ==========")
    print(ols.summary())

    # Report 95% confidence intervals to quantify uncertainty in
    # the estimated regression coefficients.
    print("\nOLS coefficient uncertainty (95% confidence intervals):")
    print(
        coefficient_confidence_intervals(ols)
        .round(6)
        .to_string()
    )

    # Calculate in-sample errors using the residuals from the fitted
    # OLS model. These describe how well the model fits these same
    # measurements; they are not independent predictive validation.
    mae = float(np.mean(np.abs(ols.resid)))
    rmse = float(np.sqrt(np.mean(ols.resid ** 2)))

    print("\nOLS model performance (in-sample; not independent predictive validation):")
    print(f"MAE:  {mae:.3f} mV")
    print(f"RMSE: {rmse:.3f} mV")
    print(f"R²:   {ols.rsquared:.5f}")

    # Test whether including the interaction term improves the OLS
    # model relative to the reduced model without that term.
    print("\n========== MODEL COMPARISON ==========")
    print("Full OLS model with interaction vs. reduced model without interaction")
    print(f"F-statistic: {comparison['f_statistic']:.3f}")
    print(f"p-value:     {comparison['p_value']:.5f}")

    # Conventional OLS inference assumes independent residuals.
    # Measurements from the same sensor may be correlated, so OLS
    # uncertainty estimates and p-values should be interpreted cautiously.
    print("\nNote: conventional OLS uncertainty assumes independent residuals.")
    print("Measurements are clustered by sensor, so interpret OLS p-values cautiously.")

    # Fit a mixed-effects model with sensor-specific random intercepts.
    # This accounts for baseline differences between sensors instead
    # of treating every measurement as fully independent.
    print("\n========== MIXED-EFFECTS MODEL ==========")

    try:
        mixed = fit_mixed(df)

        # Check convergence before interpreting the model.
        if mixed.converged:
            print("Mixed-effects model converged.")
        else:
            print(
                "WARNING: The mixed-effects model did not converge. "
                "Its estimates are shown for diagnosis only."
            )

        print(mixed.summary())

        # Estimate between-sensor variance and measurement-level residual
        # variance. These describe different sources of variability.
        sensor_variance = float(mixed.cov_re.iloc[0, 0])
        residual_variance = float(mixed.scale)

        print(f"\nEstimated sensor variance: {sensor_variance:.4f} mV²")
        print(f"Estimated residual variance: {residual_variance:.4f} mV²")

        # Report the interaction p-value only when the model converged
        # and the expected interaction coefficient is present.
        interaction_name = "concentration_centered:light_centered"

        if mixed.converged and interaction_name in mixed.pvalues:
            print(
                "Mixed-model interaction p-value: "
                f"{mixed.pvalues[interaction_name]:.5f}"
            )
        else:
            print(
                "Mixed-model interaction p-value should not be interpreted "
                "because the fit did not converge or the coefficient was absent."
            )

        # Generate mixed-model diagnostic plots only for a converged fit.
        if mixed.converged:
            plot_residuals(mixed, figure_dir, "mixed")
            plot_residual_qq(mixed, figure_dir, "mixed")

    except Exception as exc:
        # A mixed-model fitting failure should not prevent the OLS
        # analysis and its diagnostic figures from being produced.
        print(
            "Mixed-effects model could not be fitted reliably: "
            f"{type(exc).__name__}: {exc}"
        )
        print("The OLS analysis and sensor-response plots remain available.")

    # Generate diagnostic figures for the OLS model and the sensor data.
    # The response plot summarizes sensor output across experimental
    # conditions; residual plots help reveal patterns not explained
    # by the fitted regression model.
    plot_sensor_response(df, figure_dir)
    plot_residuals(ols, figure_dir, "ols")
    plot_residual_qq(ols, figure_dir, "ols")

    print(f"\nFigures saved in: {figure_dir.resolve()}")
    print(f"Analysis complete for {experiment_id}.")


if __name__ == "__main__":
    # Allow the user to select a saved experiment from the terminal.
    # For example:
    # python -m src.analyze_data --experiment experiment_002
    # Use 'latest' to analyze the most recently saved experiment.
    parser = argparse.ArgumentParser(
        description="Analyze a saved sensor experiment."
    )
    parser.add_argument(
        "--experiment",
        default="latest",
        help="Experiment ID (e.g. experiment_002) or 'latest' (default).",
    )

    args = parser.parse_args()
    main(args.experiment)
