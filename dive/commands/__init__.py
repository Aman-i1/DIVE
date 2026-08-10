"""Implementations behind each CLI command.

Kept separate from :mod:`dive.cli` so that the click layer stays a thin
declaration of the interface, and so each command can be imported lazily -
``dive --help`` should not pay the cost of importing scikit-learn.
"""
