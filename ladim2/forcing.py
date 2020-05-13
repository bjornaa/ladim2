import sys
import os
import importlib

"""Wrapper around selected orcing classes"""

def Forcing(**args):

    """Returns an instance of the Forcing class in args["module"]"""

    sys.path.insert(0, "/home/bjorn/ladim2/ladim2")
    sys.path.insert(0, os.getcwd())

    F = importlib.import_module(args['module']).Forcing
    return F(**args)
