import numpy as np


class Bounds:

    def __init__(self, generation_table, seeking_sex, target_sex, limits):
        """
        Compute bounds lol

        :param seeking_sex:
        :param target_sex: if -1, target the entire generation. else target
            sex 0 (females) or 1 (males)
        :param limits:
        """
        if seeking_sex == -1:
            self.seeking_index = np.arange(generation_table.cols.filled_rows)
        else:
            self.seeking_index = generation_table.cols.get_sex_index(
                seeking_sex)
        if target_sex == -1:
            self.target_index = np.arange(generation_table.cols.filled_rows)
        else:
            self.target_index = generation_table.cols.get_sex_index(target_sex)
        seeking_x = generation_table.cols.x[self.seeking_index]
        target_x = generation_table.cols.x[self.target_index]
        x_limits = seeking_x[:, np.newaxis] + limits
        self.bounds = np.searchsorted(target_x, x_limits)

    def __len__(self):
        return len(self.bounds)

    def get_bound_pops(self):
        """
        Compute the number of organisms captured by each bound

        :return:
        """
        return self.bounds[:, 1] - self.bounds[:, 0]