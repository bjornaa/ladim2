from pathlib import Path

from ladim.model import init_module

grid_module = Path(__file__).parent / "grid0.py"


def test_local():
    """Test the use of a local plug-in grid module"""

    # module argument must be a string (could be improved)
    config = dict(module=str(grid_module))
    g = init_module("grid", config, dict())
    assert g.metric(10, 20) == 1
    assert g.ingrid(10, 20)
