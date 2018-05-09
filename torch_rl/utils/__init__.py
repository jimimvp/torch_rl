from .utils import *

try:
    from .mpi_running_mean_std import *

except ImportError as e:
    from .running_mean_std import *