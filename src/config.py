"""Improvement: after finishing the model, I cleaned up the project structure,
first my creating a file: config.py to centralize all the parameters """
from dataclasses import dataclass


@dataclass
class SensorParams:
    # Expected sensor response, in mV
    baseline: float = 100.0
    b_conc: float = 8.0
    b_light: float = 0.01
    b_inter: float = 0.0002

    # Sources of measurement variation
    noise_std: float = 2.5
    sensor_offset_std: float = 3.0

    # None gives a linear light response.
    # A positive value enables a saturating response.
    light_scale: float | None = None


CONCENTRATIONS = [0, 2.5, 5, 7.5, 10]  # mM
LIGHT_LEVELS = [0, 100, 300, 600]      # lux

N_SENSORS = 8
N_REPS = 10
RANDOM_SEED = 42  #Reproducability, in the app 

DATA_DIR = "data/experiments"
