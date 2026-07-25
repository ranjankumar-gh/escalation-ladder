"""A simulated, fully deterministic incident environment.

No network, no wall-clock reads, no `random`. Every value is either a literal or derived
from a SHA-256 of its inputs, so the examples in the book produce identical numbers on
every machine and in every run -- years after publication.
"""
