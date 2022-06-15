.. configuration:

:mod:`configuration` --- LADiM Configuration
============================================

.. module:: configuration
   :synopsis: LADiM configuration

LADiM's configuration system uses the :mod:`pyyaml` package to read the
configuration file. This allows comments, nested keywords,
and flexibility in the sequence of keywords, missing or extra keyword,

The configuration procedure makes a dictionary. It does not quite match the structure of
the yaml configuration file as some of the values are derived or default.
Presently the dictionary is somewhat inconsistent to provide backwards compatibility.
Future versions will continue to separate the configuration info into separate
directories for the gridforce, ibm and output modules.

version [optional]
  Version number for the configuration format. Te version descried here is 2.0 

time
  start_time
    Start time for simulation,  [numpy.datetime64]
  stop_time
    Stop time for simulation,   [numpy.datetime64]
  reference_time  [optional]
    Reference time for simulation,   [numpy.datetime64]
    Used in the units attribute in the netCDF output file.
    The default value is the start_time

grid:
  module: [optional]
    If not present, it is contained in the forcing module below
    Configuration items required by the grid

forcing
  module
    Name of forcing module
    Must be in pythons `sys.path`` or be given by file path
  Configuration items required by the forcing module

release
  release_file
    Name of file with particle release schedule
  continuous: 
    True or False for discrete release
  release_frequency
    Needed for continuous release, format: [value, unit] 
  release_variables: optional
    Column names for the release file, overriden by header line in the release file

state [optional]
  State variables and their default values if needed.

ibm  [optional]
  module
    path to IBM-module

tracker
  advection [optional?]
    Advection method
    EF: Euler Forward, RK2: Runge-Kutta 2nd order, RK4 Runge-Kutta 4th order
  diffusion [optional]
    Horizontal diffusion coeffient in m^2/s 
    If missing or zero, there is enforced diffusion

output
  filename
    name of output file or name pattern for multifile output
  output_period:
    time intervall between output, format [value, unit]
  numrec: optional
    number of records per output file. Default = 0 for no output file splitting
  particle:
    list of names of particle variables that should be written
  instance:
    list of names of instance variables that should be written
  
    





particle_release_file
  Name of particle release file
output_file
  Name of output file or template for sequence of output files
start
  Simulation start "cold" or "warm"
warm_start_file
  Name of warm start file (if needed)
dt
  Model time step in seconds [int]
simulation_time
  Simulation tile in seconds [int]
numsteps
  Number of time steps [int]
gridforce
  Gridforce module with configuration
  Dictionary of information to the gridforce module
input_file
  Name of input file or template for sequence of input files
ibm_forcing
  List of extra forcing variables beside velocity
ibm
  IBM module with configuration
ibm_variables:
  List of variables needed by the IBM module
ibm_module
  Path to the IBM module
release_type
  Type of particle release, "discrete" or "continuous".
release_format
  List of variables provided during particle release
release_dtype
  Dictionary with name -> type for the release variables
particle_variables
  Names of particle variables among the release variables
output_format
  NetCDF format for the output file
skip_initial
  Logical switch for skipping output of initial field
output_numrec
  Number of time records per output file, zero means no output splitting
output_period
  Hours between output [int]
num_output
  Number of output time records
output_particle
  Particle variables included in output
output_instance
  Instance variables included in output
nc_attributes
  mapping: variable -> dictionary of netcdf attributes
advection
  Advection scheme, "EF" = Euler Forward, "RK2" = Runge-Kutta order 2,
  "RK4" = Runge-Kutta order 4
diffusion
  Logical switch for horizontal random walk diffusion
diffusion_coefficient
  Diffusion coefficient, constant [m/s**2]
