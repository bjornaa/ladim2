from ladim2 import configure


class Test_load_module:
    def test_can_load_numpy_main_module(self):
        numpy_module = configure.load_module('numpy')
        assert numpy_module.zeros(3).tolist() == [0, 0, 0]

    def test_can_load_numpy_linalg_module(self):
        linalg_module = configure.load_module('numpy.linalg')
        assert linalg_module.det([[1, 2], [0, 4]]) == 4

    def test_can_load_named_module(self):
        fname = configure.__file__
        assert fname.endswith('.py')
        module_object = configure.load_module(fname)
        assert hasattr(module_object, 'load_module')
