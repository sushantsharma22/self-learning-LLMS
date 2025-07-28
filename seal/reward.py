"""
reward.py
=========
Simple “should we accept the candidate?” helper.
"""

def should_accept(old_metric: float, new_metric: float, threshold: float) -> bool:
    """
    Accept if the improvement is ≥ `threshold`.
    Guarantees negative improvements never pass through.
    """
    return max(new_metric - old_metric, 0.0) >= threshold
