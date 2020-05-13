import sys
import os
import importlib


def Grid(**args):

    sys.path.insert(0, "/home/bjorn/ladim2/ladim2")
    sys.path.insert(0, os.getcwd())

    module = args['module']

    G = importlib.import_module(module).Grid
    return G(**args)
