"""Small shared helper functions."""


def safe_list(value):
    """Return a list for common collection inputs, otherwise an empty list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]
