import numpy as np

import pandas

"""
General Ideas

-more efficient vector operations in c++?? especially normal pdf stuff
    some testing will need to be done

"""

"""
dealing with eliminating organisms killed by environmental fitness:
-masking? (complex)~ boolean mask in flag?
-deletion (annoying and possibly intensive--copying)

"""

"""
Challenges in a fully vectorized (columnized?) table:
-saving the vectors as an array without having to create a big array
-truncation and extension are more difficult (look at tskit for help though)

"""

class ColSchema:

    pass


class Cols:
    """
    Takes the place of the large np 2d array.
    """

    def __init__(self, col_names, col_arrs, max_rows, filled_rows):
        """

        :param col_names:
        :type col_names: list
        :param col_arrs:
        :type col_arrs: list
        """
        self.col_names = col_names
        self.max_rows = max_rows
        self.filled_rows = filled_rows
        # initialize potential columns
        self.id = None
        self.parent_ids = None
        self.time = None
        self.xloc = None
        self.sex = None
        self.alleles = None
        self.flag = None
        # initialize actual columns
        for arr, name in zip(col_arrs, col_names):
            setattr(self, name, arr)

    @classmethod
    def initialize(cls, col_schema, max_rows):
        """
        Initialize empty Cols with all possible columns

        :param max_rows:
        :return:
        """
        col_names = [x for x in col_schema]
        col_arrs = []
        for col in col_schema:
            dict = col_schema[col]
            if dict["cols"] == 1:
                shape = max_rows
            else:
                shape = (max_rows, dict["cols"])
            col_arrs.append(np.zeros(shape, dtype=dict["dtype"]))
        filled_rows = 0
        return cls(col_names, col_arrs, max_rows, filled_rows)

    def __len__(self):
        """
        :return:
        """
        return self.max_rows

    def truncate(self):
        pass

    def append(self):
        pass


class Table:
    """
    Superclass of PedigreeTable GenerationTable etc
    """

    def __init__(self, cols, parameters):
        self.cols = cols
        self.parameters = parameters

    def __len__(self):
        """
        :return: total number of rows
        """
        return self.cols.max_rows

    @property
    def filled_rows(self):
        """
        :return: number of filled rows; first unfilled index
        """
        return self.cols.filled_rows

    @property
    def max_rows(self):
        """
        :return: total number of rows
        """
        return self.cols.max_rows

    def __getitem__(self, idx):
        pass

    def __setitem__(self, idx, value):
        pass

    def truncate(self, max_rows):
        """
        Remove all rows above index max_rows
        :param max_rows:
        :return:
        """
        self.cols.truncate(max_rows)



class PedigreeTable(Table):

    col_schema = {
        "id": {"dtype": np.int32,
               "cols": 1},
        "parent_ids": {"dtype": np.int32,
                       "cols": 2},
        "time": {"dtype": np.int32,
                 "cols": 1},
        "xloc": {"dtype": np.float32,
                 "cols": 1},
        "sex": {"dtype": np.uint8,
                "cols": 1},
        "alleles": {"dtype": np.uint8,
                    "cols": 4},
        "flag":  {"dtype": np.uint8,
                  "cols": 1},
    }

    col_names = [
        "id",
        "parent_ids",
        "time",
        "xloc",
        "sex",
        "alleles",
        "flag"
    ]

    def __init__(self, cols=None, parameters=None):
        if cols is None:
            pass
        super().__init__(cols, parameters)

    @classmethod
    def initialize(cls, max_rows, parameters):
        cols = Cols.initialize(cls.col_schema, max_rows)
        return cls(cols, parameters)



class GenerationTable(Table):

    col_schema = {
        "id": {"dtype": np.int32,
               "cols": 1},
        "parent_ids": {"dtype": np.int32,
                       "cols": 2},
        "time": {"dtype": np.int32,
                 "cols": 1},
        "xloc": {"dtype": np.float32,
                 "cols": 1},
        "sex": {"dtype": np.uint8,
                "cols": 1},
        "alleles": {"dtype": np.uint8,
                    "cols": 4},
        "flag":  {"dtype": np.uint8,
                  "cols": 1},
    }

    col_names = [
        "id",
        "parent_ids",
        "time",
        "xloc",
        "sex",
        "alleles",
        "flag"
    ]

    def __init__(self, cols=None, parameters=None):
        if cols is None:
            pass
        super().__init__(cols, parameters)

    @classmethod
    def initialize_founders(cls, parameters):
        """
        Initialize the founding generation
        """
        max_rows = parameters.N
        cols = Cols.initialize(cls.col_schema, max_rows)
        cols.id = np.arange(max_rows)
        cols.parent_ids[:, :] = -1
        cols.time = parameters.g
        return cls(cols, parameters)



class SubGenerationTable(Table):

    pass



















































































