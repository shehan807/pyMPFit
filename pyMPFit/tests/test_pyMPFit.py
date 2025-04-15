"""
Unit and regression test for the pyMPFit package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import pyMPFit


def test_pyMPFit_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pyMPFit" in sys.modules
