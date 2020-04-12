import pandas as pd


class Release:
    def __init__(self, filename, variables):

        # dtypes = list[variables.values()]

        self.df = pd.read_csv(
            filename, names=list(variables.keys()), delim_whitespace=True
        )

        self.total_num_particles = len(self.df)
