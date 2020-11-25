from ladim2.grid import init_grid


def test_local():
    """Test the use of a local plug-in grid module"""

    config = dict(module="grid0")
    g = init_grid(**config)
    assert g.metric(10, 20) == 1
    assert g.ingrid(10, 20) is True
