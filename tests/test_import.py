"""Test Snap Package Template."""

import snap_package_template


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(snap_package_template.__name__, str)
