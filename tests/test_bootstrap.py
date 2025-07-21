"""
Test suite for the snap_package_template module.

This module contains asynchronous tests for functions within the
bootstrap_python_package, specifically focusing on the `some_function` function.

The tests are written using the pytest framework.

Functions
---------
test_some_variable_to_test()
    Asynchronously tests that `some_function` returns the expected value.
"""

from snap_package_template import some_function


async def test_some_variable_to_test():
    """
    Test that `some_function` returns the expected string.

    This asynchronous test checks whether the `some_function` from the
    bootstrap_python_package module returns the string "some_variable_to_test".

    Raises
    ------
    AssertionError
        If the function does not return the expected string.
    """
    assert some_function() == "some_variable_to_test"
