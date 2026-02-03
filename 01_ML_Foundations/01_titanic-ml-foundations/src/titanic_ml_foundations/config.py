from __future__ import annotations  # for future compatibility
from datetime import datetime

# Define directories
OUTPUTS_DIR = "outputs"
FIGURES_DIR = "figures"  # outputs directory for figures
REPORTS_DIR = "reports"  # outputs directory for reports


# Constants
RANDOM_STATE = 6
TEST_SIZE = 0.2
N_SPLITS_CV = 5

# Model hyperparameters LR
MAX_ITER_LOGREG = 1000
# Model hyperparameters RF
N_ESTIMATORS_RF = 300