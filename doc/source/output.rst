Output
======

The particle distributions are written to NetCDF files. The file name, output frequency, included variables and their attributes are governed by the configuration file.

:index:`Output format`
----------------------

This is a not backwards compatible modification of the old LADiM format. The
fundamental data structure is the same, so scripts should be easy to modify.
The change is done to follow the CF-standard as far as possible, and to
increase flexibility.

For particle tracking, the CF-standard defines a format for "Indexed ragged
array representation of trajectories". This is not suitable for our purpose
since we are more interested in the geographical distribution of particles at a
given time than the individual trajectories. Chris Baker at NOAA has an
interesting discussion on this topic and a suggestion at `github
<https://github.com/NOAA-ORR-ERD/nc_particles/blob/master/
nc_particle_standard.md>`_. The new LADiM format is closely related to this
suggestion.

The format uses an indexed ragged array representation, suitable for both
version 3 and 4 of the NetCDF standard. The dimensions are ``time``,
``particle`` and ``particle_instance``. The particle dimension indexes the
individual particles, while the particle_instance indexes the instances i.e. a
particle at a given time. The number of times is defined by the length of the
simulation and the frequency of output, both are determined by the
configuration. The number of particles is given by the sum of the
multiplicities in the particle release file, also known in advance. The number
of instances is more uncertain as particles may be removed from the computation
by leaving the area, stranding, or becoming inactive for biological reasons.
The particle_instance dimension is therefore the single unlimited dimension allowed in the NetCDF 3 format.

The indirect indexing is given by the variable ``particle_count(time)``. The
particle distribution at timestep ``n`` (counting from initial time n=0) can be
retrieved by the following python code:

.. code-block:: python

  nc = Dataset(particle_file)
  particle_count = nc.variables['particle_count'][:n+1]
  start = np.sum(particle_count[:n])
  count = particle_count[n]
  X = nc.variables['X'][start:start+count]

Note that some languages uses 1-based indexing (MATLAB, fortran by default). In
MATLAB the code is:

.. code-block:: matlab

  % Should be checked by someone who knows matlab
  particle_count = ncread(particle_file, 'particle_count', 1, n+1)
  start = 1 + sum(particle_count(1:n))
  count = particle_count(n+1)
  X = ncread(particle_file, 'X', start, count)

Particle identifier
-------------------

The particle identifier, ``pid`` should always be present in the output file.
It is a particle number, starting with 0 and increasing as the particles are
released. The ``pid`` follows the particle and is not reused if the particle
becomes inactive.  In particular, :samp:`max(pid) + 1` is the total number of
particles involved so far in the simulation and may be larger than the number
of active particles at any given time. It also has the property that if a
particle is released before another particle, it has lower ``pid``.

The particle identifiers at a given time frame has the following properties,

* :samp:`pid[start:start+count]` is a sorted integer array.
* :samp:`pid[start+p] >= p`
  with equality if and only if all earlier particles are alive at the time
  frame.

These properties can be used to extract the individual trajectories, if needed.



Example CDL
-----------

NetCDF has a text representation, Common Data Language, abbreviated as CDL.
Here is an example CDL, produced by :command:`ncdump -h`:

.. code-block:: none

  netcdf out {
  dimensions:
        particle = 72000 ;
        particle_instance = UNLIMITED ; // (540000 currently)
        time = 13 ;

  variables:
        double time(time) ;
              time:long_name = "time" ;
              time:standard_name = "time" ;
              time:units = "seconds since 2015-04-01T00:00:00.000000" ;
        long particle_count(time) ;
              particle_count:long_name = "number of particles in a given timestep" ;
              particle_count:ragged_row_count = "particle count at nth timestep" ;
        double release_time(particle) ;
              release_time:units = "seconds since 2015-04-01T00:00:00" ;
              release_time:long_name = "particle release time" ;
        long farmid(particle) ;
              farmid:long_name = "fish farm location number" ;
        long pid(particle_instance) ;
              pid:long_name = "particle identifier" ;
        float X(particle_instance) ;
              X:long_name = "particle X-coordinate" ;
        float Y(particle_instance) ;
              Y:long_name = "particle Y-coordinate" ;
        float Z(particle_instance) ;
              Z:standard_name = "depth_below_surface" ;
              Z:positive = "down" ;
              Z:units = "m" ;
              Z:long_name = "particle depth" ;
        float super(particle_instance) ;
              super:long_name = "number of individuals in instance" ;
        float age(particle_instance) ;
              age:standard_name = "integral_of_sea_water_temperature_wrt_time" ;
              age:units = "Celcius days" ;
              age:long_name = "particle age in degree-days" ;

  // global attributes:
        :Conventions = "CF-1.5" ;
        :institution = "Institute of Marine Research" ;
        :source = "Lagrangian Advection and Diffusion Model, python version" ;
        :history = "Created by pyladim" ;
        :date = "2017-02-15" ;
  }


:index:`Split output`
---------------------

For long simulations, the output file may become large and difficult to handle.
Version 1.1 adds the possibility of split output. This is activated by adding
the keyword ``numrec`` to the output section in the configuration file. This
will split the output file after every numrec record. If the ``output_file``
has the value :file:`out.nc`, the actual files are named :file:`out_0000,nc`,
:file:`out_0001.nc`, ... .

:index:`Restart`
----------------

Version 1.1 also adds the  of :index:`warm start`, most typically a restart. In
this mode LADiM starts with an existing particle distribution, the last time
instance in an output NetCDF file.

This mode is triggered by the specification of a ``warm_start_file`` in the
configuration. As the initial distribution already exist on file, it is by
default not rewritten.

Restart and split output works nicely together.  Suppose ``numrec`` is present
and equal in both config files. If the warm start file is nam ed
:file:`out_0030.nc` and is complete, the new files will start at
:file:`out_0031.nc` and continue as in the original simulation. With no
diffusion and unchanged settings, the new files should be identical to the
original.

.. seealso::

  Module :mod:`output`
    Documentation of the :mod:`output` module.
