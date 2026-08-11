"""Pangu-prefixed public entry for the SSG pretraining architecture."""

from .ssg import SSG


class PanguSSG(SSG):
    """Backward-compatible Pangu namespace for SSG."""
