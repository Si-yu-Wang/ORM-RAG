import numpy as np

def advanced_calculate_confidence_entropy(labels, lambda_rate=0.2, alpha=1e-5, beta=10.0, gamma=1.0):
    """
    Calculates a confidence score based on the entropy and trend of label consistency, 
    using exponential weighting and Laplace smoothing.
    Parameters
    ----------
    labels : list
        A list of labels (categorical or numerical) to evaluate. Must contain at least 2 elements.
    lambda_rate : float, optional
        The exponential decay rate for weighting recent labels more heavily (default: 0.2).
    alpha : float, optional
        Laplace smoothing parameter to avoid zero probabilities (default: 1e-5).
    beta : float, optional
        Controls the steepness of the sigmoid function mapping entropy/trend to confidence (default: 10.0).
    gamma : float, optional
        Scaling factor for the trend component in the confidence calculation (default: 1.0).
    Returns
    -------
    confidence : float
        A confidence score in the range (0, 1), where higher values indicate greater consistency among labels.
    Raises
    ------
    ValueError
        If the input `labels` contains fewer than 2 elements.
    Usage
    -----
    >>> labels = ['A', 'A', 'B', 'A']
    >>> confidence = advanced_calculate_confidence_entropy(labels)
    >>> print(confidence)
    0.73  # Example output
    Notes
    -----
    - The function applies exponential weighting to recent labels, emphasizing their importance.
    - Entropy is computed to measure uncertainty, and a trend factor captures consecutive label consistency.
    - The final confidence is mapped via a sigmoid function, balancing entropy and trend.
    """
    if len(labels) < 2:
        raise ValueError("Labels must contain at least 2 values.")
    
    n = len(labels)
    first_label = labels[0]
    
    weights = np.array([np.exp(-lambda_rate * i) for i in range(n)])
    weights = weights / np.sum(weights)  # 归一化
    
    matches = np.array([1 if label == first_label else 0 for label in labels])
    weighted_match_sum = np.sum(matches * weights)
    total_weight = np.sum(weights)
    p_match = (weighted_match_sum + alpha) / (total_weight + 2 * alpha)
    p_nonmatch = 1 - p_match
    entropy = - (p_match * np.log2(p_match) + p_nonmatch * np.log2(p_nonmatch))
    
    trend = 0.0
    if n > 1:
        trend_weights = weights[1:]
        trend_matches = [1 if labels[i] == labels[i-1] else 0 for i in range(1, n)]
        trend = np.sum(trend_weights * trend_matches * matches[1:]) / np.sum(trend_weights)
    confidence = 1 / (1 + np.exp(beta * (entropy - gamma * trend)))
    return confidence
