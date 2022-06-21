.. .. author:: Bjørn Ådlandsvik <bjorn@imr.no>

LADiM configuration
===================

The LADiM model system is highly configurable using a separate configuration file. The
goal is that a user should not have to touch the code, every necessary aspect should be
customizable by the configuration file.

The name of the configuration file is given as a command line argument to the
main ladim script. If the name is missing, a default name of :file:`ladim.yaml`
is supposed.

It is a goal to provide sane defaults in the configuration file, so that
enhancements of the configuration setup do not break existing configuration
files.

The configuration file format is self-describing and can easily be modified by the user.
The format uses a subset of  `yaml (YAML Ain't Markup Language) <https:yaml.org>`_, but
does not require extensive knowledge of the ``yaml`` standard. In fact, the format fits
the simpler `StrictYAML <https://hitchdev.com/strictyaml/>`_ subset of ``yaml`` with
flow-style entries allowed.

.. note::

  The indentation in a yaml file is mandatory, it is part of the syntax. In
  particular note that the indentation has to be done by **spaces**, **not tabs**.
  Reasonable editors will recognize yaml-files by the extension and will
  automatically produce spaces if you hit the tab key.

.. seealso::
  Module :mod:`configuration`
    Documentation of the :mod:`configuration` module

An example configuration file
-----------------------------

Below is an example configuration file, :file:`examples/line/ladim.yaml`.

.. literalinclude:: ../../examples/line/ladim.yaml
  :language: yaml