""" Finally, after confirming that the simulation and data loader work correctly,
this file implements all the data analysis needed to draw meaningful conclusions 
from the generated data.

Improvement here is that there's a dedicated module for the statistical analysis, instead of
loading and analyzing the data all in one file. Here, The goal is to estimate the 
effects of concentration and illumination on the sensor output while accounting for
the fact that measurement from the same sensor are not fully independent. Changes are: 
I implemented both ordinary least squares (OLS) regression and mixed-effects regression,
which allows persistent differences between sensors to be considered. The module also 
provides confidence intervals and model comparison tools to quantify uncertainty and 
investigate whether the interaction between concentration and illumination should be 
included in the model."""


import warnings

import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import f


def prepare_data(df):
    """Center predictors to improve numerical conditioning."""
    required = {
        "concentration_mM",
        "light_lux",
        "output_mV",
        "sensor_id",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data["concentration_centered"] = (
        data["concentration_mM"] - data["concentration_mM"].mean()
    )
    data["light_centered"] = (
        data["light_lux"] - data["light_lux"].mean()
    )

    return data


def _formula(include_interaction=True):
    """Build the regression formula."""
    if include_interaction:
        return "output_mV ~ concentration_centered * light_centered"

    return "output_mV ~ concentration_centered + light_centered"


def fit_ols(df, include_interaction=True):
    """Fit an ordinary least-squares regression model."""
    data = prepare_data(df)

    return smf.ols(
        formula=_formula(include_interaction),
        data=data,
    ).fit()


def fit_mixed(df, include_interaction=True, maxiter=2000, verbose=True):
    """Fit a random-intercept mixed model, trying alternative optimizers."""
    data = prepare_data(df)

    if data["sensor_id"].nunique() < 2:
        raise ValueError(
            "Mixed-effects regression requires at least two sensors."
        )

    model = smf.mixedlm(
        formula=_formula(include_interaction),
        data=data,
        groups=data["sensor_id"],
        re_formula="1",
    )

    methods = ("lbfgs", "bfgs", "cg", "powell", "nm")
    attempts = []
    candidates = []

    for method in methods:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(
                    reml=False,
                    method=method,
                    maxiter=maxiter,
                    disp=False,
                )

            messages = [str(w.message) for w in caught]
            variance = float(result.cov_re.iloc[0, 0])
            fixed_effects = np.asarray(result.fe_params, dtype=float)
            residual_variance = float(result.scale)
            log_likelihood = float(result.llf)

            usable = (
                bool(result.converged)
                and np.all(np.isfinite(fixed_effects))
                and np.isfinite(variance)
                and variance >= 0
                and np.isfinite(residual_variance)
                and residual_variance > 0
                and np.isfinite(log_likelihood)
            )

            attempts.append(
                (method, usable, messages, variance)
            )

            if usable:
                # Prefer non-boundary fits, then compare their likelihoods.
                candidates.append(
                    (
                        variance > 1e-8,
                        log_likelihood,
                        method,
                        result,
                        messages,
                    )
                )

        except Exception as exc:
            attempts.append(
                (
                    method,
                    False,
                    [f"{type(exc).__name__}: {exc}"],
                    np.nan,
                )
            )

    if candidates:
        non_boundary = [
            candidate for candidate in candidates if candidate[0]
        ]
        pool = non_boundary if non_boundary else candidates

        _, _, method, result, messages = max(
            pool,
            key=lambda candidate: candidate[1],
        )

        variance = float(result.cov_re.iloc[0, 0])

        if verbose:
            if method != methods[0]:
                print(f"MixedLM used fallback optimizer: {method}")

            if variance <= 1e-8:
                print(
                    "WARNING: Random-intercept variance is approximately "
                    "zero. Interpret the mixed-model results cautiously."
                )

            for message in messages:
                if any(
                    word in message.lower()
                    for word in (
                        "converge",
                        "singular",
                        "boundary",
                        "gradient",
                    )
                ):
                    print(f"MixedLM warning ({method}): {message}")

        return result

    if verbose:
        print("WARNING: No usable converged mixed-effects model was found.")

        for method, usable, messages, variance in attempts:
            print(
                f"  {method}: {'usable' if usable else 'failed'} "
                f"(random variance={variance})"
            )
            for message in messages:
                print(f"    {message}")

    raise RuntimeError(
        "Mixed-effects model did not produce a usable converged fit."
    )


def coefficient_confidence_intervals(model, alpha=0.05):
    """Return confidence intervals for fitted model coefficients."""
    return model.conf_int(alpha=alpha).rename(
        columns={0: "lower", 1: "upper"}
    )


def compare_ols_models(df):
    """Compare OLS models with and without the interaction term."""
    reduced = fit_ols(df, include_interaction=False)
    full = fit_ols(df, include_interaction=True)

    degrees_freedom = int(full.df_model - reduced.df_model)

    if degrees_freedom <= 0:
        raise ValueError(
            "The full model must contain additional terms."
        )

    numerator = (
        (reduced.ssr - full.ssr) / degrees_freedom
    )
    denominator = full.ssr / full.df_resid

    f_statistic = numerator / denominator
    p_value = f.sf(
        f_statistic,
        degrees_freedom,
        full.df_resid,
    )

    return {
        "reduced_model": reduced,
        "full_model": full,
        "f_statistic": f_statistic,
        "p_value": p_value,
    }
