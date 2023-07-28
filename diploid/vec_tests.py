import numpy as np


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


class Table:

    column_names = []

    def __init__(self, table_dict):
        self.table = table


class PedigreeTableRow:

    pass


class PedigreeTable(Table):
    """
    id : int32

    parent_ids : int32

    sex : bool?

    x : float32?

    t : int32?

    genotypes : u1?

    flag : bool?

    """

    column_names = [
        "id",
        "parent_ids"
        "sex",
        "x",
        "t",
        "genotypes",
        "flag"
    ]

    def __init__(self, table):
        super().__init__(table)






























