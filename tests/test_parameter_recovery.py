import numpy as np
import pandas as pd
import pytest

from src.config import SensorParams
from src.models import fit_ols
from src.simulate import simulate


def test_simulation_is_reproducible():
    """Identical parameters and seed should produce identical data."""
    params = SensorParams()

    first = simulate(params=params, n_sensors=4, n_reps=3, seed=123)
    second = simulate(params=params, n_sensors=4, n_reps=3, seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_simulation_has_expected_number_of_measurements():
    """Every sensor receives every concentration/light combination."""
    params = SensorParams()

    df = simulate(params=params, n_sensors=4, n_reps=3, seed=123)

    expected_rows = (
        4
        * len([0, 2.5, 5, 7.5, 10])
        * len([0, 100, 300, 600])
        * 3
    )

    assert len(df) == expected_rows
    assert df["sensor_id"].nunique() == 4


def test_ols_recovers_known_parameters_without_noise():
    """
    With zero measurement noise and zero sensor offsets,
    OLS should recover the simulated linear model coefficients.
    """
    params = SensorParams(
        baseline=100.0,
        b_conc=8.0,
        b_light=0.01,
        b_inter=0.0002,
        noise_std=0.0,
        sensor_offset_std=0.0,
    )

    df = simulate(
        params=params,
        n_sensors=4,
        n_reps=2,
        seed=123,
    )

    model = fit_ols(df)

    mean_concentration = df["concentration_mM"].mean()
    mean_light = df["light_lux"].mean()

    # Because predictors are centered, the main-effect
    # coefficients include the interaction evaluated at
    # the mean value of the other predictor.
    expected_concentration_slope = (
        params.b_conc + params.b_inter * mean_light
    )
    expected_light_slope = (
        params.b_light + params.b_inter * mean_concentration
    )

    assert model.params["Intercept"] == pytest.approx(
        params.baseline
        + params.b_conc * mean_concentration
        + params.b_light * mean_light
        + params.b_inter * mean_concentration * mean_light,
        abs=1e-8,
    )

    assert model.params["concentration_centered"] == pytest.approx(
        expected_concentration_slope, abs=1e-8
    )

    assert model.params["light_centered"] == pytest.approx(
        expected_light_slope, abs=1e-8
    )

    assert model.params[
        "concentration_centered:light_centered"
    ] == pytest.approx(params.b_inter, abs=1e-8)

    assert np.sqrt(np.mean(model.resid**2)) < 1e-8
