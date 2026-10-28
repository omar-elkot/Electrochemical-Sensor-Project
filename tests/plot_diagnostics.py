
from src.loader import load_experiment
from src.models import fit_ols
from src.models import fit_mixed
from src.plots import (
    plot_sensor_response,
    plot_residuals,
    plot_residual_qq,
)


df = load_experiment()
ols = fit_ols(df)

plot_sensor_response(df)
plot_residuals(ols, model_name="ols")
plot_residual_qq(ols, model_name="ols")

mixed = fit_mixed(df)

plot_residuals(mixed, model_name="mixed")
plot_residual_qq(mixed, model_name="mixed")

print("\nAll visualization tests completed.")
