import numpy as np
from scipy import stats
import math


def _to_probability(log_odds: float) -> float:
    """Convert log odds to probability."""
    return 1 / (1 + np.exp(-log_odds))


def _calculate_uncertainty(current_posterior, data_points):
    """
    Calculate 95% confidence interval for the posterior probability.
    
    Args:
        current_posterior (float): The current posterior log-odds
        data_points (list): List of dictionaries containing l_plus and l_minus values
        
    Returns:
        tuple: (lower_bound, upper_bound) - the 95% CI bounds
    """
    # Convert current posterior log-odds to probability
    p = _to_probability(current_posterior)
    
    # Calculate total weight of evidence from Bayes factors
    total_evidence = sum(
        abs(math.exp(dp['l_plus'] - dp['l_minus']) - 1)
        for dp in data_points
    )
    
    if total_evidence < 1e-6:
        return (0.0, 1.0)  # Default CI for effectively no data
    
    # Each Bayes factor represents the weight of evidence
    # The concentration parameter of our Beta should reflect this
    concentration = total_evidence
    
    # Calculate Beta parameters to maintain the mean at p
    alpha = concentration * p
    beta = concentration * (1 - p)
    
    # Calculate 95% confidence interval
    ci_low, ci_high = stats.beta.interval(0.95, alpha, beta)
    
    # Clip to [0, 1]
    ci_low = max(0.0, min(1.0, ci_low))
    ci_high = max(0.0, min(1.0, ci_high))
    
    return ci_low, ci_high
