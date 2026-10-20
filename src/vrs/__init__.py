"""voice-role-splitter: turn a folder of character voice lines into per-role, per-tone folders.

Pipeline: embeddings from independent speaker backends -> cluster -> human names the
clusters -> calibrated top-k re-scoring of the leftovers -> human spot-check -> export.
The design assumes the model is wrong about *who* is speaking and only good at
"these two clips sound alike"; see docs/METHOD.md for the measurements behind that.
"""
__version__ = "0.2.0"

from .config import Config, ConfigError    # noqa: F401
