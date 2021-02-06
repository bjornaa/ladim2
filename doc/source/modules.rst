LADiM Modules
=============

This chapter describes briefly the present implementation
of LADiM. The code is modular and flexible with classes
that can be modified or replaced for different purposes.

The descriptions tries the requirements of the classes
from the rest of the system, hopefully being helpful for
developing alternative classes for use systems.

The code may be developed towards base classes, where alternatives
can inherit from a common base class. Presently, alternatives must
be written separately.

.. figure:: ladim_ecosystem.png
   :width: 500

   The LADiM "eco"-system; the connections between the
   modules.

.. toctree::
   :maxdepth: 2

   main_module.rst
   configuration_module.rst
   gridforce.rst
   release_module.rst
   state.rst
   tracker.rst
   IBM_module.rst
   output_module.rst
