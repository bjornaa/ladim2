import pandas as pd


class Release:
    def __init__(self, filename, dtype=None, other_variables=None):

        if dtype is None:
            dtype = dict()
        if other_variables is None:
            other_variables = dict()

        # Read the release file as a pandas dataframe
        self.df = pd.read_csv(
            filename,
            delim_whitespace=True,
            index_col="release_time",
            parse_dates=True,
            dtype=dtype,
        )

        for var, value in other_variables.items():
            self.df[var] = value

        self.total_num_particles = len(self.df)
