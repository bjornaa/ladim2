from ladim2.grid import makegrid


def test_local():
    """Test the use of a local plug-in grid module"""

    config = dict(module="grid0")
    g = makegrid(**config)
    assert g.metric(10, 20) == 1
    assert g.ingrid(10, 20) is True
