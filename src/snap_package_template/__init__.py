"""
Snap Package Template.

This is a template for the Snap Python package, designed for easy starting and
structuring of Python projects. This package includes basic versioning and a
simple function example.

Attributes
----------
__version__ : str
    The current version of the package.
__version_tuple__ : tuple
    The current version of the package as a tuple.
"""

__version__ = "0.0.0"
__version_tuple__ = (0, 0, 0)


def some_function() -> str:
    """
    Demonstrate the structure of a Python function.

    This function returns a test string.

    Returns
    -------
    str
        A string containing "some_variable_to_test".
    """
    return "some_variable_to_test"
