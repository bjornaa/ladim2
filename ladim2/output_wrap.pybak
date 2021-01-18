import sys
import os
import importlib

"""Wrapper around selected output classes"""


def Output(**args):

    """Returns an instance of the Output class in args["module"]"""

    sys.path.insert(0, "/home/bjorn/ladim2/ladim2")
    sys.path.insert(0, os.getcwd())

    F = importlib.import_module(args["module"]).Output
    return F(**args)
