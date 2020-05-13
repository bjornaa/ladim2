
from .state import State
from .grid import Grid
from .timer import Timer
from .forcing import Forcing
from .tracker import Tracker
from .release import ParticleReleaser
from .output import Output
from .configure import configure

# Limitation, presently only instantaneous particle release

def main(configuration_file):
    """Main function for complete particle tracking model"""

    # ----------------
    # Configuration
    # ----------------

    config = configure(configuration_file)


    # -------------------
    # Initialization
    # -------------------

    state = State(**config["state"])
    timer = Timer(**config["time_control"])
    grid = Grid(**config["grid"])
    force = Forcing(grid=grid, timer=timer, **config["forcing"])
    tracker = Tracker(**config["tracker"])
    release = ParticleReleaser(time_control=timer, **config["release"])
    output = Output(state=state, timer=timer, **config["output"])

    # --------------------------
    # Initial particle release
    # --------------------------

    # Skal automatisere tilpasning til state-variablene
    # Også initiering av variable som ikke er i release-filen
    # X0 er et eksempel.
    print("Initial particle release")
    V = next(release)
    # TODO: Simplify release
    ## next provides pid, this is handled by state itself
    # V = V.drop(columns='pid')
    state.append(**V)

    # --------------
    # Time loop
    # --------------

    print("Time loop")
    for step in range(timer.Nsteps):
        tracker.update(state, grid=grid, force=force)
        if step % output.frequency == 0:
            output.write(step)

    # --------------
    # Finalisation
    # --------------

    print("Cleaning up")
    output.save_particle_variables()
    output.close()
