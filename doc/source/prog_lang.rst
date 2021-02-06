Programming language
====================

The LADiM code is written in `python <https://www.python.org>`_, more
specific it requires python 3.6 or newer.

Dependencies
------------

In addition to the python standard library, the model is based on the numerical
package `numpy <http:www.numpy.org>`_ and the NetCDF library `netcdf4-python
<http://unidata.github.io/netcdf4-python>`_. The yaml package
`pyyaml <http://pyyaml.org>`_ is used for the configuration, while the particle
release module depend on the data analysis package
`pandas <http://pandas.pydata.org>`_. All these packages are available in the
`anaconda <https://www.continuum.io/anaconda-overview>`_ bundle.

The postprocessing library, :ref:`postladim<postladim>`, uses the `xarray
<http://xarray.pydata.org>`_ framework. In addition the examples use the plotting
package `matplotlib <http://matplotlib.org>`_ for visualisation.

Testing is done using the `pytest <http://doc.pytest.org>`_-framework.

This documentation uses the `Sphinx <http://www.sphinx-doc.org>`_
documentation system to generate html and pdf documentation. To produce the
pdf version a decent `LaTeX <https://www.latex-project.org>`_ implementation
is needed.