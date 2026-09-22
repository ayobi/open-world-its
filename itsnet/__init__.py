"""itsnet — a learned ITS flank detector.

A base-wise neural segmenter for the rDNA operon, intended to replace or back
up the HMMER profile search used by ITSx / ITSxpress / ITSxRust.
"""
__version__ = "0.1.0"

from .labels import N_STATES, STATES, STATE_IDX  # noqa: F401
