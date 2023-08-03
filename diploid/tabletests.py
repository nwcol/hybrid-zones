import numpy as np

import os

from diploid.parameters import Params

"""
challenges
-selection for sampling in pedigree
-sort by x     : (
-sort by id
"""


class Constants:

    genotypes = np.array([[1, 1, 1, 1],
                          [1, 1, 1, 2],
                          [1, 1, 2, 2],
                          [1, 2, 1, 1],
                          [1, 2, 1, 2],
                          [1, 2, 2, 2],
                          [2, 2, 1, 1],
                          [2, 2, 1, 2],
                          [2, 2, 2, 2]], dtype=np.uint8)


class Columns:

    _col_names = ["ID",
                  "maternal_ID",
                  "paternal_ID",
                  "time",
                  "sex"]

    def __init__(self, ID, maternal_ID, paternal_ID, time, sex, filled_rows,
                 max_rows, **kwargs):
        """
        All arrays must match max_rows in length.

        :param params:
        :param ID:
        :param maternal_ID:
        :param paternal_ID:
        :param time:
        :param sex:
        :param max_rows:
        :param filled_rows:
        :param kwargs:
        """
        self.max_rows = max_rows
        self.filled_rows = filled_rows
        self.ID = ID
        self.maternal_ID = maternal_ID
        self.paternal_ID = paternal_ID
        self.time = time
        self.sex = sex
        self.col_names = [col_name for col_name in self._col_names]
        if "x" in kwargs:
            self.x = kwargs["x"]
            self.col_names.append("x")
        else:
            self.x = None
        if "alleles" in kwargs:
            self.alleles = kwargs["alleles"]
            self.col_names.append("alleles")
        else:
            self.alleles = None
        if "genotype_ID" in kwargs:
            self.genotype_ID = kwargs["genotype_ID"]
            self.col_names.append("genotype_ID")
        else:
            self.genotype_ID = None
        if "flag" in kwargs:
            self.flag = kwargs["flag"]
            self.col_names.append("flag")
        else:
            self.flag = None
        lengths = [len(getattr(self, col)) for col in self.col_names]
        if len(set(lengths)) > 1:
            raise AttributeError("column length mismatch")
        if lengths[0] != self.max_rows:
            raise AttributeError("max_rows mismatch")
        if self.filled_rows > self.max_rows:
            raise ValueError("filled_rows exceeds max_rows")

    def __repr__(self):
        out = (f"Cols w/ {self.filled_rows} of {self.max_rows} rows filled "
               f"in {len(self.col_names)} columns")
        return out

    def __str__(self, n=10):
        """

        :param n: number of entries to print from either end
        :return:
        """
        if len(self) < 2 * n:
            n1 = len(self)
            n2 = 0
            omit = False
        else:
            n1 = n2 = n
            omit = True
        header, spacer = self.format_header()
        _out = [spacer, header, spacer]
        for i in np.arange(n1):
            _out.append(self.format_row(i))
        if omit:
            _out.append(". . .")
            omitted = self.max_rows - n1 - n2
            _out.append(str(omitted) + " rows omitted for brevity")
            _out.append(". . .")
        if len(self) > 1:
            for i in np.arange(self.max_rows-n2, self.max_rows):
                _out.append(self.format_row(i))
        _out.append(spacer)
        out = "\n".join(_out)
        return out

    def format_row(self, index):
        """
        Return a formatted string representation of one row in the table
        all max-length values should have buffer | xx...xx | of one space

        :param row: the row index of interest
        :return:
        """
        lengths = self.get_col_widths()
        _row = [""]
        for col_name, length in zip(self.col_names, lengths):
            entry = getattr(self, col_name)[index]
            if col_name == "alleles":
                entry = ' '.join([str(a) for a in entry])
            if col_name == "x":
                entry = np.round(entry, 5)
            _row.append(f"{str(entry) : >{length}}")
        _row.append("")
        row = " | ".join(_row)
        return row

    def format_header(self):
        """
        Return a string of the elements in :self.col_names: formatted as a
        header for the table

        :return: a formatted header string
        """
        lengths = self.get_col_widths()
        _header = [""]
        _spacer = [""]
        for col_name, length in zip(self.col_names, lengths):
            _header.append(f"{col_name : >{length}}")
            _spacer.append("=" * length)
        _header.append("")
        _spacer.append("")
        header = " | ".join(_header)
        spacer = "=|=".join(_spacer)
        return header, spacer

    def get_col_widths(self):
        """
        for formatting

        :return:
        """
        lengths = []
        for col_name in self.col_names:
            if col_name == "alleles":
                lengths.append(10)
            if col_name == "x":
                lengths.append(7)
            else:
                lengths.append(
                    max(len(col_name),
                        len(str(np.max(getattr(self, col_name))))))
        return lengths

    def __len__(self):
        return self.max_rows

    def __getitem__(self, index):
        """
        Adapted from tskit basetable class. Return a subset of the pedigree
        using a single integer index, list or array of indices, slice,
        or boolean mask

        :param index:
        :return:
        """
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError("index out of bounds")
            index = [index]
        elif isinstance(index, slice):
            index = range(*index.indices(len(self)))
        else:
            index = np.asarray(index)
            if index.dtype == np.bool_:
                if len(index) != len(self):
                    raise IndexError(
                        "boolean index must be same length as table")
                index = np.flatnonzero(index)
        ID = self.ID[index]
        maternal_ID = self.maternal_ID[index]
        paternal_ID = self.paternal_ID[index]
        time = self.time[index]
        sex = self.sex[index]
        n = np.size(index)
        kwargs = dict()
        for column in self.col_names:
            if column not in Columns._col_names:
                kwargs[column] = getattr(self, column)[index]
        return Columns(ID, maternal_ID, paternal_ID, time, sex, n, n, **kwargs)

    def __setitem__(self, key, value):
        """
        Set rows at index "key" equal to value. key must be an integer, list,
        slice or array of integers, and its length must exactly match the
        number of rows in value. value is a Columns instance with max_rows <
        the max_rows of this instance

        all columns in THIS instance must exist in value, but all columns in
        value need not exist in THIS (eg value may contain x, but THIS does
        not; then x is ignored)

        :param key:
        :param value:
        :return:
        """
        if isinstance(key, int):
            key = [key]
        elif isinstance(key, slice):
            key = range(*key.indices(len(self)))
        else:
            key = np.asarray(key)
            key = np.flatnonzero(key)
        if len(key) != len(value):
            raise ValueError("length of value does not match length of key!")
        for col_name in self.col_names:
            if col_name not in value.col_names:
                raise ValueError(col_name + " not present in value!")
            own_column = getattr(self, col_name)
            val_column = getattr(value, col_name)
            own_column[key] = val_column
            setattr(self, col_name, own_column)

    def append_generation(self, gen_cols):
        """

        :return:
        """
        ## no support for overflow rn
        n = len(gen_cols)
        self[self.filled_rows:self.filled_rows + n] = gen_cols
        self.filled_rows += n

    def apply_ID(self):
        """
        Add ids to the array

        :return:
        """
        self.ID += np.arange(self.max_rows)

    def sort_by_x(self):
        """
        Sort organisms by order of increasing x position

        :return:
        """
        if "x" in self.col_names:
            index = np.argsort(self.x)
            for column in self.col_names:
                setattr(self, column, getattr(self, column)[index])
        else:
            raise AttributeError("no x column has been declared")

    def get_sex_subset(self, target_sex):
        """
        Return a new Columns instance holding those organisms of sex
        :target_sex:

        :param target_sex:
        :return:
        """
        mask = self.sex == target_sex
        return self[mask]

    def get_sex_index(self, target_sex):
        """
        Return the indices of organisms with sex :target_sex:

        :param target_sex:
        :return:
        """
        return np.where(self.sex == target_sex)[0]

    def get_subpop_size(self, **kwargs):
        """
        Return the number of organisms with character defined in **kwargs
        using the format column=character.

        example
        >cols.get_subpop_size(sex=0)
        4985

        :param kwargs:
        :return:
        """
        index = np.arange(len(self))
        for arg in kwargs:
            new = np.where(getattr(self, arg) == kwargs[arg])[0]
            index = np.intersect1d(index, new)
        return len(index)

    def get_subpop(self, **kwargs):
        """
        Return the number of organisms with character defined in **kwargs
        using the format column=character.

        example
        >cols.get_subpop_size(sex=0)
        4985

        :param kwargs:
        :return:
        """
        index = np.arange(len(self))
        for arg in kwargs:
            new = np.where(getattr(self, arg) == kwargs[arg])[0]
            index = np.intersect1d(index, new)
        return self[index]

    def save_txt(self):
        pass

    def save_ped(self):
        """
        Save as a .ped format file
        """
        pass

    def save_npz(self, filename):
        """
        Save as a .npz format file

        :param filename:
        :return:
        """
        arg_dict = dict()
        for col_name in self.col_names:
            arg_dict[col_name] = getattr(self, col_name)
        np.savez(filename, **arg_dict)

    @classmethod
    def load_npz(cls, filename):
        archive = np.load(filename)
        col_names = archive.files
        ID = archive["ID"]
        maternal_ID = archive["maternal_ID"]
        paternal_ID = archive["paternal_ID"]
        time = archive["time"]
        sex = archive["sex"]
        n = archive.max_header_size
        kwargs = dict()
        for col_name in col_names:
            if col_name not in cls._col_names:
                kwargs[col_name] = archive[col_name]
        return cls(ID, maternal_ID, paternal_ID, time, sex, n, n, **kwargs)


"""
operations to figure out

X-sorting by x or id
X-printing individual row
X-return individual row as array?
-
X-id'ing (ez)
-saving as txt
-saving as np format

-start adding all applicable stuff

"""


class Table:

    def __init__(self, params, ID, maternal_id, paternal_id, time, sex,
                 filled_rows, max_rows, **kwargs):
        self.cols = Columns(ID, maternal_id, paternal_id, time, sex,
                            filled_rows, max_rows, **kwargs)
        self.params = params


class GenerationTable(Table):

    def __init__(self, params, ID, maternal_id, paternal_id, time, sex,
                 filled_rows, max_rows, **kwargs):
        super().__init__(params, ID, maternal_id, paternal_id, time, sex,
                         filled_rows, max_rows, **kwargs)
        self.sort_by_x()
        self.apply_ID()


    @classmethod
    def get_founding(cls, params):
        """create the founding generation"""
        n = params.N
        ID = np.full(n, 0, dtype=np.int32)
        time = np.full(n, params.g, dtype=np.int32)
        maternal_ID = np.full(n, -1, dtype=np.int32)
        paternal_ID = np.full(n, -1, dtype=np.int32)
        sex = np.random.choice(np.array([0, 1], dtype=np.uint8), size=n)
        alleles_ = []
        x_ = []
        for i, genotype in enumerate(Constants.genotypes):
            n_ = params.subpop_n[i]
            if n_ > 0:
                alleles_.append(np.repeat(genotype[np.newaxis, :], n_, axis=0))
                lower, upper = params.subpop_lims[i]
                x_.append(np.random.uniform(lower, upper, n_))
        alleles = np.vstack(alleles_)
        x = np.concatenate(x_)
        return cls(params, ID, maternal_ID, paternal_ID, time, sex, n, n, x=x,
                   alleles=alleles)

    def sort_by_x(self):
        self.cols.sort_by_x()

    def apply_ID(self):
        self.cols.apply_ID()


class PedigreeTable(Table):

    size_factor = 1.01

    def __init__(self, params, ID, maternal_ID, paternal_ID, time, sex,
                 filled_rows, max_rows, **kwargs):
        super().__init__(params, ID, maternal_ID, paternal_ID, time, sex,
                         filled_rows, max_rows, **kwargs)
        self.time_now = params.g

    @classmethod
    def initialize(cls, params, full=True):
        filled_rows = 0
        max_rows = int(params.K * (params.g + 1) * cls.size_factor)
        ID = np.zeros(max_rows, dtype=np.int32)
        time = np.zeros(max_rows, dtype=np.int32)
        maternal_ID = np.zeros(max_rows, dtype=np.int32)
        paternal_ID = np.zeros(max_rows, dtype=np.int32)
        sex = np.zeros(max_rows, dtype=np.uint8)
        if full:
            kwargs = {"x" : np.zeros(max_rows, dtype=np.float32),
                      "alleles" : np.zeros((max_rows, 4), dtype=np.uint32),
                      "flag" : np.zeros(max_rows, dtype=np.uint32)}
        else:
            kwargs = dict()
        return cls(params, ID, maternal_ID, paternal_ID, time, sex,
                   filled_rows, max_rows, **kwargs)



params = Params(10_000, 10, 0.1)

gen = GenerationTable.get_founding(params)

cols = gen.cols
