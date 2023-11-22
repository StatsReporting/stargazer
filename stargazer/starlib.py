"""
This file contains utilities that (differently from those in utils.py) are only
meant for internal use.
"""

def _extract_feature(obj, feature):
    """
    Just return obj.feature if present and None otherwise.
    """
    try:
        return getattr(obj, feature)
    except AttributeError:
        return None
