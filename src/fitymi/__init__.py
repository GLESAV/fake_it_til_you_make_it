"""Controlled study of synthetic-image training for acne severity classification.

The organising question is not "does synthetic data help" (it does, as augmentation)
but "can it substitute, and at what exchange rate". Every module here exists to keep
that comparison honest: matched budgets, a sealed test set, and controls that bound
how much of an observed effect could be an artefact.
"""

__version__ = "0.1.0"
