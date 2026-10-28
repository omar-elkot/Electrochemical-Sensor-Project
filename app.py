#Interactive dashboard

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    SensorParams,
    CONCENTRATIONS,
    LIGHT_LEVELS,
    N_SENSORS,
    N_REPS,
    RANDOM_SEED,
)
from src.simulate import simulate
from src.models import fit_ols, fit_mixed


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"


@st.cache_data(show_spinner=False)
def load_validation_csv(filename, results_dir):
    """Load a committed validation result; cache the parsed table."""
    path = Path(results_dir) / filename
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


st.set_page_config(
    page_title="Electrochemical Sensor Lab",
    page_icon="⚡",
    layout="wide",
)

# ---------- Visual style ----------
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        padding: 18px;
        border-radius: 12px;
    }
    h1, h2, h3 {letter-spacing: -0.4px;}
</style>
""", unsafe_allow_html=True)


# ---------- Sidebar: experiment controls ----------
st.sidebar.title("⚡ SENSOR LAB")
st.sidebar.caption("SIMULATION & ANALYSIS WORKSPACE")
st.sidebar.divider()

st.sidebar.subheader("Experiment design")

n_sensors = st.sidebar.slider(
    "Number of sensors", 2, 16, N_SENSORS
)
n_reps = st.sidebar.slider(
    "Replicates per condition", 2, 20, N_REPS
)
noise = st.sidebar.slider(
    "Measurement noise (mV)", 0.1, 8.0, 2.5, 0.1
)
offset_sd = st.sidebar.slider(
    "Sensor-to-sensor variation (mV)", 0.0, 8.0, 3.0, 0.5
)

st.sidebar.subheader("Sensor response")

b_conc = st.sidebar.slider(
    "Concentration sensitivity (mV/mM)",
    1.0, 15.0, 8.0, 0.5
)
b_light = st.sidebar.slider(
    "Illumination effect (mV/lux)",
    0.0, 0.03, 0.01, 0.001,
    format="%.3f",
)
b_inter = st.sidebar.slider(
    "Concentration × light interaction",
    -0.0005, 0.0008, 0.0002, 0.00005,
    format="%.5f",
)

saturating = st.sidebar.checkbox(
    "Use saturating illumination response", False
)
light_scale = None
if saturating:
    light_scale = st.sidebar.slider(
        "Illumination saturation scale (lux)",
        50, 2000, 200, 50
    )
    st.sidebar.caption(
        "This generates a saturating response, but the fitted model remains linear. "
        "Residuals may reveal the mismatch; visibility depends on noise and settings."
    )

if "seed" not in st.session_state:
    st.session_state.seed = RANDOM_SEED

if st.sidebar.button("↻ Generate new experiment", use_container_width=True):
    st.session_state.seed += 1

st.sidebar.caption(
    "All generated measurements are simulated, not experimental data."
)


# ---------- Generate measurements ----------
params = replace(
    SensorParams(),
    b_conc=b_conc,
    b_light=b_light,
    b_inter=b_inter,
    noise_std=noise,
    sensor_offset_std=offset_sd,
    light_scale=light_scale,
)

df = simulate(
    params=params,
    n_sensors=n_sensors,
    n_reps=n_reps,
    seed=st.session_state.seed,
)

# ---------- Header ----------
st.title("Electrochemical Sensor Lab")
st.markdown(
    "### Measurement characterization · Calibration · Uncertainty"
)
st.caption(
    f"SIMULATED EXPERIMENT  /  Seed {st.session_state.seed}  /  "
    "Python statistical analysis"
)

st.divider()

# ---------- Fit models ----------
ols = fit_ols(df)
interaction_name = "concentration_centered:light_centered"
interaction_p = float(ols.pvalues.get(interaction_name, np.nan))

try:
    mixed = fit_mixed(df, verbose=False)
    mixed_ok = bool(mixed.converged)
    mixed_p = float(mixed.pvalues.get(interaction_name, np.nan))
    sensor_variance = float(mixed.cov_re.iloc[0, 0])
except Exception:
    mixed = None
    mixed_ok = False
    mixed_p = np.nan
    sensor_variance = np.nan

predicted = ols.predict()
residuals = df["output_mV"] - predicted
rmse = float(np.sqrt(np.mean(residuals**2)))
r2 = float(ols.rsquared)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Measurements", f"{len(df):,}")
m2.metric("Sensors", f"{df['sensor_id'].nunique()}")
m3.metric("OLS R²", f"{r2:.4f}")
m4.metric("OLS RMSE", f"{rmse:.2f} mV")
headline_p = mixed_p if mixed_ok and np.isfinite(mixed_p) else interaction_p
headline_label = (
    "Interaction p-value (MixedLM)"
    if mixed_ok and np.isfinite(mixed_p)
    else "Interaction p-value (OLS fallback)"
)
m5.metric(
    headline_label,
    f"{headline_p:.3g}" if np.isfinite(headline_p) else "N/A",
)

# ---------- Main views ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Calibration & response",
    "🔬 Sensor variability",
    "🧪 Statistical diagnostics",
    "✅ Model Validation",
])

with tab1:
    left, right = st.columns([1.5, 1])

    with left:
        st.subheader("Concentration-response curves")

        # Match src/plots.py: average replicates within each sensor-condition
        # first, then show the standard error across sensor means.
        by_sensor = (
            df.groupby(["sensor_id", "light_lux", "concentration_mM"])["output_mV"]
            .mean()
            .reset_index(name="sensor_mean_mV")
        )
        summary = (
            by_sensor.groupby(["light_lux", "concentration_mM"])["sensor_mean_mV"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_output"})
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["count"])

        fig = px.line(
            summary,
            x="concentration_mM",
            y="mean_output",
            color="light_lux",
            markers=True,
            error_y="sem",
            labels={
                "concentration_mM": "Concentration (mM)",
                "mean_output": "Mean sensor output (mV)",
                "sem": "Standard error across sensor means (mV)",
                "light_lux": "Illumination (lux)",
            },
            template="plotly_dark",
        )
        fig.update_layout(
            height=470,
            legend_title="Illumination",
            margin=dict(l=10, r=10, t=25, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Measurement distribution")

        fig = px.scatter(
            df,
            x="light_lux",
            y="output_mV",
            color="concentration_mM",
            opacity=0.55,
            labels={
                "light_lux": "Illumination (lux)",
                "output_mV": "Output (mV)",
                "concentration_mM": "Concentration (mM)",
            },
            template="plotly_dark",
        )
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Calibration data")
    st.caption(
        "Error bars show the standard error across sensor-level means. "
        "Replicates are averaged within each sensor and condition first."
    )
    st.dataframe(summary.round(3), use_container_width=True)

with tab2:
    left, right = st.columns(2)

    with left:
        st.subheader("Sensor-to-sensor response")
        sensor_summary = (
            df.groupby("sensor_id")
            .agg(
                mean_output=("output_mV", "mean"),
                sd_output=("output_mV", "std"),
            )
            .reset_index()
        )

        fig = px.box(
            df,
            x="sensor_id",
            y="output_mV",
            color="sensor_id",
            labels={
                "sensor_id": "Sensor ID",
                "output_mV": "Output (mV)",
            },
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False, height=430)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Estimated sensor offsets")
        if mixed_ok:
            random_effects = pd.DataFrame([
                {
                    "sensor_id": int(sensor_id),
                    "estimated_offset_mV": float(effects.iloc[0]),
                }
                for sensor_id, effects in mixed.random_effects.items()
            ])

            fig = px.bar(
                random_effects,
                x="sensor_id",
                y="estimated_offset_mV",
                labels={
                    "sensor_id": "Sensor ID",
                    "estimated_offset_mV": "Estimated offset (mV)",
                },
                template="plotly_dark",
            )
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Estimated vs. true sensor offsets")
            true_offsets = (
                df.groupby("sensor_id")["sensor_offset_mV"]
                .first()
                .sort_index()
                .astype(float)
            )
            estimated_offsets = pd.Series({
                int(sensor_id): float(effects.iloc[0])
                for sensor_id, effects in mixed.random_effects.items()
            }).sort_index()
            common_ids = true_offsets.index.intersection(estimated_offsets.index)
            if len(common_ids) == df["sensor_id"].nunique():
                true_centered = true_offsets.loc[common_ids] - true_offsets.loc[common_ids].mean()
                estimated_centered = (
                    estimated_offsets.loc[common_ids]
                    - estimated_offsets.loc[common_ids].mean()
                )
                offset_comparison = pd.DataFrame({
                    "True centered offset (mV)": true_centered.to_numpy(),
                    "Estimated centered offset (mV)": estimated_centered.to_numpy(),
                    "Sensor ID": common_ids,
                })
                fig = px.scatter(
                    offset_comparison,
                    x="True centered offset (mV)",
                    y="Estimated centered offset (mV)",
                    hover_data=["Sensor ID"],
                    labels={
                        "True centered offset (mV)": "True centered offset (mV)",
                        "Estimated centered offset (mV)": "Estimated centered offset (mV)",
                    },
                    template="plotly_dark",
                )
                low = float(min(offset_comparison["True centered offset (mV)"].min(),
                                offset_comparison["Estimated centered offset (mV)"].min()))
                high = float(max(offset_comparison["True centered offset (mV)"].max(),
                                 offset_comparison["Estimated centered offset (mV)"].max()))
                fig.add_shape(
                    type="line", x0=low, y0=low, x1=high, y1=high,
                    line=dict(dash="dash"),
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "True and estimated offsets are centered separately because "
                    "mixed-model random effects represent deviations from the population intercept."
                )
            else:
                st.info("Offset comparison is unavailable because sensor IDs do not match.")
        else:
            st.warning("Mixed-effects model did not converge for this run.")

    st.subheader("Sensor summary")
    st.dataframe(sensor_summary.round(3), use_container_width=True)

with tab3:
    st.subheader("Model comparison")

    comparison = pd.DataFrame({
        "Metric": [
            "OLS R²",
            "OLS adjusted R²",
            "OLS RMSE (in-sample)",
            "OLS interaction p-value",
            "Mixed-model interaction p-value",
            "Estimated sensor variance (mV²)",
        ],
        "Value": [
            f"{ols.rsquared:.5f}",
            f"{ols.rsquared_adj:.5f}",
            f"{rmse:.4f}",
            f"{interaction_p:.5g}",
            f"{mixed_p:.5g}" if np.isfinite(mixed_p) else "Unavailable",
            f"{sensor_variance:.4f}" if np.isfinite(sensor_variance) else "Unavailable",
        ],
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.caption(
        "OLS treats observations as independent for its standard errors. "
        "The mixed-effects model includes a sensor-level random intercept. "
        "Compare their interaction p-values with that distinction in mind."
    )

    st.subheader("OLS residual diagnostics")
    left, right = st.columns(2)

    with left:
        fig = px.scatter(
            x=predicted,
            y=residuals,
            labels={
                "x": "Predicted output (mV)",
                "y": "Residual (mV)",
            },
            template="plotly_dark",
            title="Residuals vs predicted output",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            x=residuals,
            nbins=35,
            labels={"x": "Residual (mV)"},
            template="plotly_dark",
            title="Residual distribution",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Download simulated measurements")
    st.download_button(
        "⬇ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="simulated_sensor_measurements.csv",
        mime="text/csv",
    )


with tab4:
    st.subheader("Scientific validation")
    st.caption(
        "Monte Carlo validation using synthetic data. These results do not "
        "establish performance on physical sensors."
    )

    parameter_df = load_validation_csv("parameter_validation.csv", str(RESULTS_DIR))
    comparison_df = load_validation_csv("model_comparison_validation.csv", str(RESULTS_DIR))
    false_positive_df = load_validation_csv("false_positive_validation.csv", str(RESULTS_DIR))
    power_df = load_validation_csv("interaction_power_analysis.csv", str(RESULTS_DIR))

    # 1. Parameter recovery
    st.markdown("### 1. Parameter recovery")

    required_parameter_columns = {
        "true_interaction", "estimated_interaction",
        "ci_lower", "ci_upper", "ci_width"
    }
    if (
        parameter_df is not None
        and not parameter_df.empty
        and required_parameter_columns.issubset(parameter_df.columns)
    ):
        errors = (
            parameter_df["estimated_interaction"]
            - parameter_df["true_interaction"]
        )
        true_value = float(parameter_df["true_interaction"].mean())
        mean_estimate = float(parameter_df["estimated_interaction"].mean())
        bias = float(errors.mean())
        recovery_rmse = float(np.sqrt(np.mean(errors**2)))
        coverage = float((
            (parameter_df["ci_lower"] <= parameter_df["true_interaction"])
            & (parameter_df["ci_upper"] >= parameter_df["true_interaction"])
        ).mean() * 100)
        mean_width = float(parameter_df["ci_width"].mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Simulations", f"{len(parameter_df):,}")
        c2.metric("True interaction", f"{true_value:.6g}")
        c3.metric("Mean estimate", f"{mean_estimate:.6g}")
        c4.metric("Estimation RMSE", f"{recovery_rmse:.6g}")

        st.write(f"Mean bias: `{bias:.6g}`")
        st.write(f"95% confidence-interval coverage: `{coverage:.1f}%`")
        st.write(f"Mean confidence-interval width: `{mean_width:.6g}`")

        recovery_plot = px.histogram(
            parameter_df,
            x="estimated_interaction",
            nbins=30,
            title="Distribution of estimated interaction coefficients",
            template="plotly_dark",
        )
        recovery_plot.add_vline(
            x=true_value,
            line_dash="dash",
            line_color="white",
            annotation_text="True value",
        )
        recovery_plot.update_layout(height=380)
        st.plotly_chart(recovery_plot, use_container_width=True)
    else:
        st.warning(
            "parameter_validation.csv is missing, empty, or does not contain "
            "the expected columns."
        )

    st.divider()

    # 2. OLS versus mixed-effects model
    st.markdown("### 2. OLS versus MixedLM")

    required_comparison_columns = {
        "model", "bias", "covered", "ci_width"
    }
    if (
        comparison_df is not None
        and not comparison_df.empty
        and required_comparison_columns.issubset(comparison_df.columns)
    ):
        model_summary = (
            comparison_df.groupby("model")
            .agg(
                simulations=("bias", "count"),
                mean_bias=("bias", "mean"),
                estimation_rmse=("bias", lambda x: np.sqrt(np.mean(x**2))),
                ci_coverage=("covered", "mean"),
                mean_ci_width=("ci_width", "mean"),
            )
            .reset_index()
        )
        model_summary["ci_coverage"] *= 100
        model_summary = model_summary.rename(columns={
            "model": "Model",
            "simulations": "Simulations",
            "mean_bias": "Mean bias",
            "estimation_rmse": "Estimation RMSE",
            "ci_coverage": "95% CI coverage (%)",
            "mean_ci_width": "Mean CI width",
        })

        st.dataframe(
            model_summary.round(6),
            hide_index=True,
            use_container_width=True,
        )

        coverage_plot = px.bar(
            model_summary,
            x="Model",
            y="95% CI coverage (%)",
            title="Confidence-interval coverage by model",
            template="plotly_dark",
        )
        coverage_plot.add_hline(
            y=95,
            line_dash="dash",
            line_color="white",
            annotation_text="Nominal 95%",
        )
        coverage_plot.update_layout(height=350)
        st.plotly_chart(coverage_plot, use_container_width=True)

        st.caption(
            "Coverage is the percentage of simulated confidence intervals "
            "that contain the known interaction coefficient. Higher coverage "
            "is not automatically better if intervals are unnecessarily wide."
        )
    else:
        st.warning(
            "model_comparison_validation.csv is missing, empty, or does not "
            "contain the expected columns."
        )

    st.divider()

    # 3. False-positive validation
    st.markdown("### 3. False-positive validation")

    required_fp_columns = {"model", "false_positive"}
    if (
        false_positive_df is not None
        and not false_positive_df.empty
        and required_fp_columns.issubset(false_positive_df.columns)
    ):
        fp_summary = (
            false_positive_df.groupby("model")
            .agg(
                simulations=("false_positive", "size"),
                false_positives=("false_positive", "sum"),
                false_positive_rate=("false_positive", "mean"),
            )
            .reset_index()
        )
        fp_summary["False-positive rate (%)"] = (
            fp_summary["false_positive_rate"] * 100
        )

        st.dataframe(
            fp_summary[
                ["model", "simulations", "false_positives",
                 "False-positive rate (%)"]
            ].rename(columns={
                "model": "Model",
                "simulations": "Simulations",
                "false_positives": "False positives",
            }).round(2),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "The nominal significance level is 5%. With a finite number of "
            "simulations, observed rates naturally fluctuate around that level."
        )
    else:
        st.warning(
            "false_positive_validation.csv is missing, empty, or does not "
            "contain the expected columns."
        )

    st.divider()

    # 4. Statistical power
    st.markdown("### 4. Statistical power")

    required_power_columns = {
        "scenario", "requested_simulations", "usable_fits",
        "failed_or_unusable_fits",
        "detection_or_false_positive_rate_pct",
    }
    if (
        power_df is not None
        and not power_df.empty
        and required_power_columns.issubset(power_df.columns)
    ):
        power_display = power_df.rename(columns={
            "scenario": "Scenario",
            "requested_simulations": "Requested simulations",
            "usable_fits": "Usable fits",
            "failed_or_unusable_fits": "Failed/unusable fits",
            "detection_or_false_positive_rate_pct":
                "Detection / false-positive rate (%)",
        })

        st.dataframe(
            power_display.round(2),
            hide_index=True,
            use_container_width=True,
        )

        power_plot = px.bar(
            power_display,
            x="Scenario",
            y="Detection / false-positive rate (%)",
            text="Detection / false-positive rate (%)",
            title=(
                "Detection with an interaction vs. false positives under the null"
            ),
            template="plotly_dark",
        )
        power_plot.update_layout(height=380)
        st.plotly_chart(power_plot, use_container_width=True)

        st.caption(
            "Detection rate is estimated under the specified simulated "
            "interaction. The null rate estimates false positives when the "
            "interaction coefficient is zero. Results depend on the simulation "
            "design and noise assumptions."
        )
    else:
        st.warning(
            "interaction_power_analysis.csv is missing, empty, or does not "
            "contain the expected columns."
        )

st.divider()
st.caption(
    "Electrochemical Sensor Project · Research prototype · "
    "Simulated data only — not validated against physical sensor measurements."
)
