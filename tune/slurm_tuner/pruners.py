import numpy as np
from dataclasses import dataclass

@dataclass
class SuccessiveHalvingPruner:
    """
    min_trials: int
        The minimum number of trials to run before pruning.
    reduction_factor: int
        The percentage of all trials to prune.
        
    - Take num trails from grid search
    - In the end of the process you want the 5 most promising trials to complete
    - 
    
    """
