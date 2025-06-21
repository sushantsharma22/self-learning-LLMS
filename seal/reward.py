"""
reward.py
=========
Tiny helper: decide whether a new metric beats the threshold.
"""


def should_accept(
    old_metric: float, new_metric: float, threshold: float
) -> bool:
    """
    Return True if `new_metric` improves on `old_metric` by at least
    `threshold`.

    In the paper they use a *binary reward*: 1 if improved, else 0.
    """
    return (new_metric - old_metric) >= threshold
