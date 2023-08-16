import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import scipy.optimize as opt

import time

from hybzones import dispersal

from hybzones import fitness

from hybzones import mating

from hybzones import math_util

from hybzones import parameters

from hybzones import plot_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


"""
GENERAL TO-DO

- flux edge thing

- adding migrants to pedigrees

- test dispersal models

- fully annotate dispersal models

- fix fitness models if need be

- work on mating model

- add __repr__ etc for base table class, pay some attention to this class

- decide on final directory structure

- this should take into account the best way to run scripts on the cluster

- fix all the functions/methods which save files to make sure they flexibly
save in the correct directories

- set up interpreter script to allow task selection on the cluster

- learn how to manipulate files and directories with cmd prompt and start 
doing this more

- learn how to interact with the cluster using prompt or shell (???) ask people

- learn more about running python and package distributions


- handling parents when the upper cutoff is less than params.g in pedigree
sampling

- complex sample defines and simpler sample set objects

- tests directory setup

- make the mating model more efficient and more comprehensible
lol big task

- .ped file format

- emergency table extension 

- parse out getting/setting attributes (columns) and make sure it works right
already implemented but worth checking.

- better __repr__, __str__ for pedigree and generation tables

- when to sort by x and id for max consistency and least use

- sort out how to handle properties between columns vs tables (direct access
or access through table property?)

- columns initialization; polish up a bit

- handling zero length columns

- print dtypes below columns when __str__

- keeping track of metadata outside of tskit/msprime

- get everything set up to be able to make many slices over the same 
pedigree and run coalescence on each

- plotting death frequency in space

- handling extinction and ungraceful exits for simulation

"""


class Columns:
    """
    The core of pedigree and generation table objects, and therefore of the
    simulation.
    """

    # these are the columns essential to the function of the pedigree. they
    # are therefore mandatory to instantiate a Columns instance. time is
    # arguably not needed but its inclusion is very convenient
    _col_names = ["id",
                  "maternal_id",
                  "paternal_id",
                  "time"]

    def __init__(self, filled_rows, max_rows, **kwargs):
        """
        Constructor for the Columns class. Kwargs is used as a field for all
        column arrays, although four of them (listed in _col_names) are
        essential for the column to be initialized. This may change.
        All arrays must have a length equal to max_rows.

        :param filled_rows: index of the highest filled row. if the column
            is a pedigree being initialized, should equal 0; if a generation,
            should equal max_rows
        :type filled_rows: int
        :param max_rows: the length of the column arrays
        :type max_rows: int

        document kwargs later
        """
        # this clarifies the data types each array should have. these are
        # checked during initialization to ensure proper behavior
        types = {"id": np.int32,
                 "maternal_id": np.int32,
                 "paternal_id": np.int32,
                 "time": np.int32,
                 "sex": np.uint8,  # takes only 0, 1
                 "x": np.float32,
                 "alleles": np.uint8,  # takes only 0, 1
                 "flag": np.int8,  # takes -2, -1, 0, 1
                 "genotype_code": np.uint8}  # takes 0, 1, 2

        if filled_rows > max_rows:
            raise ValueError("filled_rows exceeds max_rows")
        self.filled_rows = filled_rows
        self.max_rows = max_rows
        self.col_names = [name for name in kwargs]
        for col_name in self.col_names:
            if kwargs[col_name].dtype != types[col_name]:
                raise TypeError(f"{col_name} has improper data type")
        for _col_name in self._col_names:
            if _col_name not in kwargs:
                raise ValueError(f"kwargs did not include column {_col_name}")

        self.id = kwargs["id"]
        self.maternal_id = kwargs["maternal_id"]
        self.paternal_id = kwargs["paternal_id"]
        self.time = kwargs["time"]
        if "sex" in kwargs:
            self.sex = kwargs["sex"]
        if "x" in kwargs:
            self.x = kwargs["x"]
        if "alleles" in kwargs:
            self.alleles = kwargs["alleles"]
        if "flag" in kwargs:
            self.flag = kwargs["flag"]
        if "genotype_code" in kwargs:
            self.genotype_code = kwargs["genotype_code"]

        lengths = [len(getattr(self, col)) for col in self.col_names]
        if len(set(lengths)) > 1:
            raise AttributeError("column length mismatched to column length")
        if lengths[0] != self.max_rows:
            raise AttributeError("max_rows mismatched to column length")

    @classmethod
    def empty(cls, max_rows, col_names):
        """
        Used by constructors. Initialize an empty Columns instance of length
        max_rows

        :param max_rows:
        :param col_names: list of column names. the 4 required columns should
            be included but aren't checked.
        :return:
        """
        kwargs = dict()
        kwargs["id"] = np.zeros(max_rows, dtype=np.int32)
        kwargs["maternal_id"] = np.zeros(max_rows, dtype=np.int32)
        kwargs["paternal_id"] = np.zeros(max_rows, dtype=np.int32)
        kwargs["time"] = np.full(max_rows, -1, dtype=np.int32)
        if "sex" in col_names:
            kwargs["sex"] = np.zeros(max_rows, dtype=np.uint8)
        if "x" in col_names:
            kwargs["x"] = np.zeros(max_rows, dtype=np.float32)
        if "alleles" in col_names:
            kwargs["alleles"] = np.zeros((max_rows, 4), dtype=np.uint8)
        if "flag" in col_names:
            kwargs["flag"] = np.full(max_rows, -10, dtype=np.int8)
        if "genotype_code" in col_names:
            kwargs["genotype_code"] = np.zeros(max_rows, dtype=np.uint8)
        filled_rows = 0
        return cls(filled_rows, max_rows, **kwargs)

    @classmethod
    def merge(cls, cols1, cols2):
        filled_rows = len(cols1) + len(cols2)
        max_rows = filled_rows
        kwargs = dict()
        for col_name in cols1.col_names:
            if col_name in cols2.col_names:
                kwargs[col_name] = np.concatenate(
                    (getattr(cols1, col_name), getattr(cols2, col_name))
                )
            else:
                raise AttributeError(f"column {col_name} is not in cols2")
        return cls(filled_rows, max_rows, **kwargs)

    def __repr__(self):
        out = (f"Cols with {self.filled_rows} filled rows of "
               f"{self.max_rows} max rows "
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

        :param index: the row index of interest
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
            _spacer.append("-" * length)
        _header.append("")
        _spacer.append("")
        header = " | ".join(_header)
        spacer = "-|-".join(_spacer)
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
        Adapted from the __getitem__ method in the tskit base table class.
        Return a new Columns instance holding a subset of this instance
        using 1. an integer, 2. a slice, 3. an array of integers (index), or
        4. a boolean mask

        example
        >cols
        Cols with 10000 filled rows of 10000 max rows in 8 columns
        >cols[10]
        Cols with 1 filled rows of 1 max rows in 8 columns
        >cols[10].id
        array([10])
        >cols[10:20]
        Cols with 10 filled rows of 10 max rows in 8 columns
        >cols[10, 20, 40, 100, 200]
        Cols with 5 filled rows of 5 max rows in 8 columns
        >cols[10, 20, 40, 100, 200].id
        array([ 10,  20,  40, 100, 200])

        :param index: the integer, slice, index or mask to access
        :type index: integer, slice, array of integers, or boolean mask
        :return: Subset of self accessed by index
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
        kwargs = dict()
        for column in self.col_names:
            kwargs[column] = getattr(self, column)[index]
        filled_rows = np.size(index)
        max_rows = np.size(index)
        return Columns(filled_rows, max_rows, **kwargs)

    def __setitem__(self, key, value):
        """
        Set rows at index "key" equal to value. key must be an integer, list,
        slice or array of integers, and its length must exactly match the
        number of rows in value. value is a Columns instance with max_rows <
        the max_rows of this instance

        all columns in THIS instance must exist in value, but all columns in
        value need not exist in THIS (e.g. value may contain x, but THIS does
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

    @property
    def parents(self):
        """
        Return a 2d array of parent ids

        :return:
        """
        return np.column_stack((self.maternal_id, self.paternal_id))

    @property
    def signal_alleles(self):
        if "alleles" not in self.col_names:
            raise AttributeError("no alleles columns exist in this instance")
        return self.alleles[:, [0, 1]]

    @property
    def preference_alleles(self):
        if "alleles" not in self.col_names:
            raise AttributeError("no alleles columns exist in this instance")
        return self.alleles[:, [2, 3]]

    @property
    def signal_sums(self):
        if "alleles" not in self.col_names:
            raise AttributeError("no alleles columns exist in this instance")
        return np.sum(self.alleles[:, [0, 1]], axis=1)

    @property
    def preference_sums(self):
        if "alleles" not in self.col_names:
            raise AttributeError("no alleles columns exist in this instance")
        return np.sum(self.alleles[:, [2, 3]], axis=1)

    @property
    def signal(self):
        return self.signal_sums - 2

    @property
    def preference(self):
        return self.preference_sums - 2

    @property
    def allele_sums(self):
        sums = np.zeros((self.filled_rows, 2), dtype=np.uint8)
        sums[:, 0] = np.sum(self.signal_alleles, axis=1)
        sums[:, 1] = np.sum(self.preference_alleles, axis=1)
        return sums

    @property
    def genotype(self):
        allele_sums = self.allele_sums
        genotype = np.zeros(self.filled_rows, dtype=np.uint8)
        for i in np.arange(Constants.n_genotypes):
            genotype[(allele_sums[:, 0] == Constants.allele_sums[i, 0])
                     & (allele_sums[:, 1] == Constants.allele_sums[i, 1])] = i
        return genotype

    def apply_id(self, i_0=0):
        """
        Add ids to the array

        :return:
        """
        self.id += np.arange(self.max_rows) + i_0

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
        return self[self.sex == target_sex]

    def get_sex_index(self, target_sex):
        """
        Return the indices of organisms with sex :target_sex:

        :param target_sex:
        :return:
        """
        return np.nonzero(self.sex == target_sex)[0]

    def get_range_index(self, range):
        """
        Return an index accessing individuals in the x range [range0, range1)

        :param range: tuple or list of length 2
        :return:
        """
        return np.nonzero((self.x >= range[0]) & (self.x < range[1]))[0]

    def get_subpop_index(self, **kwargs):
        """
        Return the index of organisms with character defined in **kwargs
        using the format column=character.

        It is not essential, but likely more efficient, to target the groups
        expected to be smallest first

        This is such a cool functionality. You are so based

        example
        >>>cols.get_subpop_index(sex=0)
        array([0, 1, 3, ... , 9878, 9879], dtype=np.int64)

        :param kwargs:
        :return:
        """
        index = np.arange(len(self))
        for arg in kwargs:
            if arg == 'x':
                if kwargs[arg][1] < kwargs[arg][0]:
                    raise ValueError("left bound exceeds right bound!")
                new = self.get_range_index(kwargs[arg])
            else:
                new = np.nonzero(getattr(self, arg) == kwargs[arg])[0]
            index = np.intersect1d(index, new)
        return index

    def get_subpop_mask(self, **kwargs):
        """
        Return a mask of organisms with the characters defined in **kwargs
        e.g. the intersection of masks for each kwarg

        """
        mask = np.full(len(self), True, dtype='bool')
        for arg in kwargs:
            if arg == 'x':
                if kwargs[arg][1] < kwargs[arg][0]:
                    raise ValueError("left bound exceeds right bound!")
                new = (self.x >= kwargs[arg][0]) & (self.x < kwargs[arg][1])
            else:
                new = getattr(self, arg) == kwargs[arg]
            mask *= new
        return mask

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
        index = self.get_subpop_index(**kwargs)
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
        index = self.get_subpop_index(**kwargs)
        return self[index]

    def truncate(self, new_max):
        """
        Replace each column array with a view of itself shortened to new_max

        :param new_max: the index to slice to
        :return:
        """
        for col_name in self.col_names:
            setattr(self, col_name, getattr(self, col_name)[:new_max])
        self.max_rows = new_max

    def reverse_truncate(self, new_min):
        """
        Replace each column array with a view of itself sliced between new_min
        and its end. max_rows is set to the new length of the columns, e.g.
        max_rows - new_min, and filled_rows is set to

        :param new_min: the index to slice from
        """
        for col_name in self.col_names:
            setattr(self, col_name, getattr(self, col_name)[new_min:])
        self.max_rows -= new_min
        self.filled_rows = self.max_rows

    def as_dict(self):
        """
        Return a dict of columns

        :return:
        """
        col_dict = {col_name: getattr(self, col_name) for col_name in
                    self.col_names}
        return col_dict

    def as_arr(self):
        """
        Return a structured array holding the Columns instance

        """
        types = [("id", "i4"), ("maternal_id ", "i4"), ("paternal_id", "i4"),
                 ("time", "i4")]
        if "sex" in self.col_names:
            types.append(("sex", "u1"))
        if "x" in self.col_names:
            types.append(("x", "f4"))
        if "alleles" in self.col_names:
            types.append(("A loc 0", 'u1'))
            types.append(("A loc 1", 'u1'))
            types.append(("B loc 0", 'u1'))
            types.append(("B loc 1", 'u1'))
        if "flag" in self.col_names:
            types.append(("flag", "i1"))
        col_names = [x[0] for x in types]
        dtype = np.dtype(types)
        arr = np.zeros(self.max_rows, dtype)
        arr["id"] = self.id
        arr["maternal_id "] = self.maternal_id
        arr["paternal_id"] = self.paternal_id
        arr["time"] = self.time
        if "sex" in self.col_names:
            arr["sex"] = self.sex
        if "x" in self.col_names:
            arr["x"] = np.round(self.x, 5)
        if "alleles" in self.col_names:
            arr["A loc 0"] = self.alleles[:, 0]
            arr["A loc 1"] = self.alleles[:, 1]
            arr["B loc 0"] = self.alleles[:, 2]
            arr["B loc 1"] = self.alleles[:, 3]
        if "flag" in self.col_names:
            arr["flag"] = self.flag
        return arr, col_names

    def save_txt(self, filename):
        """
        Get a structured array of the Columns instance and save it at
        filename

        """
        arr, col_names = self.as_arr()
        header = "".join([f"{x : >12}" for x in col_names])[3:]
        file = open(filename, 'w')
        np.savetxt(file, arr, '%11s', header=header)
        file.close()

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
        filled_rows = archive.max_header_size
        max_rows = archive.max_header_size
        kwargs = dict()
        for col_name in col_names:
            kwargs[col_name] = archive[col_name]
        return cls(filled_rows, max_rows, **kwargs)


class Table:

    def __init__(self, cols, params):
        self.cols = cols
        self.params = params

    def __len__(self):
        return len(self.cols)

    def __getitem__(self, index):
        """
        Return a copy of the subset in self.cols accessed by "index", which
        is an integer, slice, array or mask.

        The object returned is a base table instance and not a generation or
        pedigree table

        :param index:
        :return:
        """
        cols = self.cols[index]
        params = self.params
        return Table(cols, params)

    @property
    def nbytes(self):
        """
        Estimate the minimum number of bytes occupied in memory by the column
        arrays

        :return:
        """
        nbytes = 0
        col_dict = self.cols.as_dict()
        nbytes += np.sum([col.nbytes for col in col_dict.values()])
        return nbytes

    @property
    def filled_rows(self):
        return self.cols.filled_rows

    @property
    def max_rows(self):
        return self.cols.max_rows

    @property
    def time(self):
        return self.cols.time

    @property
    def x(self):
        return self.cols.x

    @property
    def alleles(self):
        return self.cols.alleles

    @property
    def allele_sums(self):
        return self.cols.allele_sums

    @property
    def genotype(self):
        return self.cols.genotype


class GenerationTable(Table):

    def __init__(self, cols, params, t):
        super().__init__(cols, params)
        self.t = t
        self.sort_by_x()

    @classmethod
    def get_founding(cls, params):
        """
        create the founding generation
        """
        n = params.N
        id = np.full(n, 0, dtype=np.int32)
        time = np.full(n, params.g, dtype=np.int32)
        maternal_id = np.full(n, -1, dtype=np.int32)
        paternal_id = np.full(n, -1, dtype=np.int32)
        sex = np.random.choice(np.array([0, 1], dtype=np.uint8), size=n)
        alleles_ = []
        x_ = []
        for i, genotype in enumerate(Constants.genotypes):
            n_ = params.subpop_n[i]
            if n_ > 0:
                alleles_.append(np.repeat(genotype[np.newaxis, :], n_, axis=0))
                low, high = params.subpop_lims[i]
                x_.append(np.random.uniform(low, high, n_).astype(np.float32))
        alleles = np.vstack(alleles_)
        x = np.concatenate(x_)
        flag = np.full(n, 1, dtype=np.int8)
        cols = Columns(n, n, id=id, maternal_id=maternal_id,
                       paternal_id=paternal_id, time=time, sex=sex, x=x,
                       alleles=alleles, flag=flag)
        cols.sort_by_x()
        cols.apply_id()
        t = params.g
        return cls(cols, params, t)

    @classmethod
    def mate(cls, parent_generation_table):
        """
        Form a new generation by mating in the previous generation

        :param parent_generation_table:
        :return:
        """
        t = parent_generation_table.t - 1
        matings = mating.Matings(parent_generation_table)
        n = matings.n
        id = np.zeros(n, dtype=np.int32)  # do later
        maternal_id = matings.abs_maternal_ids
        paternal_id = matings.abs_paternal_ids
        time = cls.get_time_col(n, t)
        sex = cls.get_random_sex(n)
        x = parent_generation_table.cols.x[matings.maternal_ids]
        alleles = matings.get_zygotes(parent_generation_table)
        flag = np.full(n, 1, dtype=np.int8)
        cols = Columns(n, n, id=id, maternal_id=maternal_id,
                       paternal_id=paternal_id, time=time, sex=sex, x=x,
                       alleles=alleles, flag=flag)
        params = parent_generation_table.params
        return cls(cols, params, t)

    @classmethod
    def merge(cls, gen_table1, gen_table2):
        if gen_table1.t != gen_table2.t:
            raise AttributeError("time mismatch between generation parts")
        t = gen_table1.t
        params = gen_table1.params
        cols = Columns.merge(gen_table1.cols, gen_table2.cols)
        return cls(cols, params, t)

    @classmethod
    def from_cols(cls, cols, params, t):
        """
        Instantiate a generation from a bare cols object

        :param cols:
        :param params:
        :param t:
        :return:
        """
        return cls(cols, params, t)

    def __repr__(self):
        return (f"GenerationTable at t = {self.t}, self.cols: \n"
                + self.cols.__repr__())

    def __str__(self):
        return (f"GenerationTable at t = {self.t}, self.cols: \n"
                + self.cols.__str__())

    @staticmethod
    def get_random_sex(n):
        return np.random.choice(np.array([0, 1], dtype=np.uint8), size=n)

    @staticmethod
    def get_time_col(n, t):
        return np.full(n, t, dtype=np.int32)

    def sort_by_x(self):
        self.cols.sort_by_x()

    def apply_ID(self):
        self.cols.apply_ID()

    @property
    def living_mask(self):
        """
        Return a mask for individuals with flag=1

        :return:
        """
        return self.cols.get_subpop_mask(flag=1)

    @property
    def living_index(self):
        """
        Return the indices of individuals with flag=1

        """
        return self.cols.get_subpop_index(flag=1)

    def set_flag(self, index, flag):
        """
        This is present only in the generation table because no other table
        should set flags

        :param index:
        :param flag:
        :return:
        """
        if "flag" not in self.cols.col_names:
            raise AttributeError("no flags column exists")
        self.cols.flag[index] = flag

    def senescence(self):
        """
        Set all individuals flagged as 1 (living) to 0 (dead)

        :return:
        """
        self.set_flag(self.living_index, 0)

    def plot(self):
        gen_arr = GenotypeArr.from_generation(self)
        fig = gen_arr.plot_density()
        return fig


class PedigreeTable(Table):

    size_factor = 1.04

    def __init__(self, cols, params, t, g):
        super().__init__(cols, params)
        self.g = g
        self.t = t

    @classmethod
    def initialize_from_params(cls, params, col_names):
        """
        Constructor for simulation pedigrees

        :param params:
        :param col_names:
        :return:
        """
        max_rows = int(params.K * (params.g + 1) * cls.size_factor)
        cols = Columns.empty(max_rows, col_names)
        g = params.g
        t = params.g
        return cls(cols, params, g, t)

    def __repr__(self):
        return (f"PedigreeTable with g = {self.g}, t = {self.t}, "
                f"{self.filled_rows} of {self.max_rows} filled")

    def __str__(self):
        return (f"PedigreeTable with g = {self.g}, t = {self.t}, \n"
                f"{self.filled_rows} of {self.max_rows} filled "
                + self.cols.__str__())

    def append_generation(self, generation_table):
        """
        Enter a generation table into the pedigree table in the lowest empty
        indices. Raises ValueError if the generation cannot fit; this breaks
        the simulation

        :return:
        """
        now_filled = self.filled_rows + len(generation_table)
        if now_filled > self.max_rows:
            raise ValueError("the pedigree table is full!")
        self.cols[self.filled_rows:now_filled] = generation_table.cols
        self.cols.filled_rows = now_filled

    def get_generation(self, t):
        """
        Create a GenerationTable instance holding all the individuals at time
        t.

        :param t:
        :return:
        """
        # this should equal the original gen object in every way
        # except for any columns which the gen object might have had which
        # weren't included in the pedigree
        mask = self.cols.time == t
        gen_cols = self.cols[mask]
        params = self.params
        return GenerationTable.from_cols(gen_cols, params, t)

    def truncate(self, new_max=None):
        """
        Reduce the length of the table Columns instance to new_max, or if no
        new_max is provided, reduce to self.cols.filled_rows

        This should return views of the original arrays in cols

        :param new_max:
        :return:
        """
        if not new_max or new_max > self.max_rows:
            new_max = self.filled_rows
        self.cols.truncate(new_max)

    def get_ancestries(self, t=0):
        """
        Compute the genealogical ancestry values of the individuals living
        at generation t
        """
        n = len(self)
        parents = self.cols.parents
        founder_index = self.cols.get_subpop_index(time=self.g)
        anc = np.zeros((n, 3), dtype=np.float32)
        anc[founder_index, :2] = parents[founder_index]
        anc[founder_index, 2] = self.cols.alleles[founder_index, 0] - 1
        gen_index = None
        for i in np.arange(self.g - 1, t - 1, -1):
            gen_index = self.cols.get_subpop_index(time=i)
            gen_parents = parents[gen_index]
            anc[gen_index, :2] = gen_parents
            anc[gen_index, 2] = np.mean([anc[gen_parents[:, 0], 2],
                                         anc[gen_parents[:, 1], 2]], axis=0)
        ancestries = anc[gen_index, 2]
        return ancestries

    def get_genotype_ancestries(self, t=0, n_bins=10):
        """
        Compute the mean genealogical ancestry value for spatial and genotype
        bins
        """
        generation = self.get_generation(t)
        ancestries = self.get_ancestries(t)
        ranges = plot_util.get_ranges(n_bins)
        arr = np.zeros((n_bins, Constants.n_genotypes))
        for geno in np.arange(Constants.n_genotypes):
            for i in np.arange(n_bins):
                xbin = ranges[i]
                index = generation.cols.get_subpop_index(x=xbin, genotype=geno)
                arr[i, geno] = np.mean(ancestries[index])

        # currently returns nans for empty groups
        return arr

    def plot_history(self, plot_int):
        """

        """
        snaps = np.arange(self.g, -1, -plot_int)
        n_figs = len(snaps)
        if n_figs in Constants.shape_dict:
            n_rows, n_cols = Constants.shape_dict[n_figs]
        else:
            n_rows = 2
            n_cols = (n_figs + 1) // 2
        plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
        figure.tight_layout(pad=3.0)
        figure.subplots_adjust(right=0.9)
        for i in np.arange(n_figs):
            t = snaps[i]
            index = np.unravel_index(i, plot_shape)
            ax = axs[index]
            genotype_arr = GenotypeArr.from_generation(self.get_generation(t))
            genotype_arr.get_subplot(ax)
        if n_figs < plot_shape[0] * plot_shape[1]:
            index = np.unravel_index(n_figs, plot_shape)
            allele_arr = AlleleArr.from_generation(self.get_generation(0))
            ax = axs[index]
            allele_arr.get_subplot(ax)
        figure.legend(["N", "Hyb"] + Constants.subpop_legend, fontsize=10,
                      loc='right', borderaxespad=0,  fancybox=False,
                      framealpha=1, edgecolor="black")
        figure.show()
        return figure


class Trial:

    def __init__(self, params, plot_int=None):
        self.run_time_0 = time.time()
        self.run_time_vec = np.zeros(params.g + 1)
        self.report_int = max(min(100, params.g // 10), 1)
        self.plot_int = plot_int
        self.figure = None
        self.complete = False
        self.g = params.g
        self.t = params.g
        self.params = params
        self.pedigree_table = PedigreeTable.initialize_from_params(params,
            col_names=["sex", "x", "alleles", "flag"])
        self.run()

    @classmethod
    def new(cls, K, g, plot_int=None, **kwargs):
        params = parameters.Params(K, g, 0.1)
        for arg in kwargs:
            setattr(params, arg, kwargs[arg])
        return cls(params, plot_int)

    def run(self):
        print("simulation initiated @ " + math_util.get_time_string())
        self.run_time_vec[self.params.g] = 0
        generation_table = GenerationTable.get_founding(self.params)
        while self.t > 0:
            generation_table = self.cycle(generation_table)
        self.pedigree_table.append_generation(generation_table)
        self.pedigree_table.truncate()
        if self.plot_int:
            self.figure = self.pedigree_table.plot_history(self.plot_int)
            self.figure.show()
        print("simulation complete")

    def cycle(self, parent_table):
        """
        Advance the simulation through a single cycle
        """
        self.update_t()
        generation_table = GenerationTable.mate(parent_table)
        parent_table.senescence()
        self.pedigree_table.append_generation(parent_table)
        dispersal.main(generation_table)
        fitness.main(generation_table)
        generation_table.cols.sort_by_x()
        generation_table.cols.apply_id(i_0=self.pedigree_table.filled_rows)
        self.report()
        return generation_table

    def update_t(self):
        self.t -= 1
        if self.t == 0:
            self.complete = True

    def report(self):
        self.run_time_vec[self.t] = time.time() - self.run_time_0
        if self.t % self.report_int == 0:
            t = self.run_time_vec[self.t]
            t_last = self.run_time_vec[self.t + self.report_int]
            mean_t = str(np.round((t - t_last) / self.report_int, 3))
            run_t = str(np.round(self.run_time_vec[self.t], 2))
            time_string = math_util.get_time_string()
            print(f"g{self.t : > 6} complete, runtime = {run_t : >8}"
                  + f" s, averaging {mean_t : >8} s/gen, @ {time_string :>8}")


class GenotypeArr:
    time_ax = 0
    space_ax = 1
    genotype_ax = 2

    def __init__(self, arr, params, t, bin_size):
        self.arr = arr
        self.params = params
        self.t = t
        self.g = params.g
        self.bin_size = bin_size

    @classmethod
    def initialize(cls, params, bin_size=0.01):
        n_bins = plot_util.get_n_bins(bin_size)
        t_dim = params.g + 1
        arr = np.zeros((t_dim, n_bins, Constants.n_genotypes), dtype=np.int32)
        return cls(arr, params, t_dim, bin_size)

    @classmethod
    def from_generation(cls, generation_table, bin_size=0.01):
        """
        Get a SubpopArr of time dimension 1, recording a single generation
        """
        bin_edges, n_bins = plot_util.get_bins(bin_size)
        arr = np.zeros((1, n_bins, Constants.n_genotypes), dtype=np.int32)
        x = generation_table.x
        genotype = generation_table.genotype
        for i in np.arange(Constants.n_genotypes):
            arr[0, :, i] = np.histogram(x[genotype == i],
                                        bins=bin_edges)[0]
        params = generation_table.params
        t = generation_table.t
        return cls(arr, params, t, bin_size)

    @classmethod
    def from_pedigree(cls, pedigree_table, bin_size=0.01):
        """Get a SubpopArr recording population densities in a Pedigree of
        time dimension pedigree.g + 1
        """
        t_dim = pedigree_table.g + 1
        n_bins = plot_util.get_n_bins(bin_size)
        arr = np.zeros((t_dim, n_bins, Constants.n_genotypes), dtype=np.int32)
        for t in np.arange(pedigree_table.g + 1):
            generation_table = pedigree_table.get_generation(t)
            arr[t, :, :] = GenotypeArr.from_generation(generation_table).arr[0]
        params = pedigree_table.params
        t = 0
        return cls(arr, params, t, bin_size)

    @classmethod
    def load_txt(cls, filename):
        file = open(filename, 'r')
        string = file.readline()
        params = parameters.Params.from_string(string)
        raw_arr = np.loadtxt(file, dtype=np.int32)
        file.close()
        shape = np.shape(raw_arr)
        t_dim = shape[0]
        n_genotypes = Constants.n_genotypes
        bin_size = shape[1] // n_genotypes
        new_shape = (t_dim, bin_size, n_genotypes)
        arr = np.reshape(raw_arr, new_shape)
        t_now = 0
        return cls(arr, params, t_now, bin_size)

    def __repr__(self):
        """
        Return a string description of the instance

        :return:
        """
        return (f"GenotypeArr of {len(self)} generations, t = {self.t}, "
                f"g = {self.g}, n organisms = {self.size}")

    def __str__(self):
        """
        Return a more detailed summary

        :return:
        """
        # write this
        return 0

    def __len__(self):
        """
        Return the number of generations recorded in the SubpopArr e.g. the
        length of the zeroth 'time' axis

        :return: length
        """
        return np.shape(self.arr)[0]

    def __getitem__(self, index):
        """
        Return the generation or generations at the times or mask designated
        by index
        """
        arr = self.arr[[index]]
        params = self.params
        bin_size = self.bin_size
        return AlleleArr(arr, params, index, bin_size)

    def enter_generation(self, generation):
        t = generation.t
        self.arr[t, :, :] = GenotypeArr.from_generation(generation).arr[0]

    @property
    def size(self):
        """
        Return the total number of organisms recorded in the array
        """
        return np.sum(self.arr)

    @property
    def generation_sizes(self):
        """
        Return a vector of population sizes for each recorded generation
        """
        return np.sum(np.sum(self.arr, axis=1), axis=1)

    @property
    def densities(self):
        """
        Return the total population densities in each generation
        """
        return np.sum(self.arr, axis=2)

    def save_txt(self, filename):
        """
        Reshape the array to be 2d with shape (generations, n_bins * n_geno.)
        and save it as a .txt file
        """
        shape = np.shape(self.arr)
        reshaped = self.arr.reshape(shape[0], shape[1] * shape[2])
        file = open(filename, 'w')
        header = str(vars(self.params))
        np.savetxt(file, reshaped, delimiter=' ', newline='\n', header=header,
                   fmt="%1.1i")
        file.close()
        print("SubpopArr saved at " + filename)

    def get_generation_size(self, t):
        """
        Return the population size at generation t
        """
        return np.sum(self.arr[t])

    def get_hybrid_densities(self, t):
        """
        Compute the sum of densities of the subpopulations with one or more
        heterozygous loci at generation t
        """
        return np.sum(self.arr[t, :, 1:8], axis=1)

    def get_densities(self, t):
        """
        Return a vector of whole population bin densities in generation t
        """
        return np.sum(self.arr[t], axis=1)

    def get_subplot(self, sub, t=0):
        """
        Plot genotype densities for a single generation on a subplot

        :param sub:
        :param t:
        :return:
        """
        b = plot_util.get_bin_mids(self.bin_size)
        n_vec = self.get_densities(t)
        sub.plot(b, n_vec, color="black", linestyle='dashed', linewidth=2,
                 marker="x")
        sub.plot(b, self.get_hybrid_densities(t), color='green',
                 linestyle='dashed', linewidth=2, marker="x")
        c = Constants.genotype_colors
        for i in np.arange(9):
            sub.plot(b, self.arr[t, :, i], color=c[i], linewidth=2, marker="x")
        y_max = self.params.K * 1.35 * self.bin_size
        n = str(self.get_generation_size(t))
        if len(self) == 1:
            time = self.t
        else:
            time = t
        title = "t = " + str(time) + " n = " + n
        plot_util.setup_space_plot(sub, y_max, "subpop density", title)

    def plot_density(self, t=0):
        """
        Make a plot of the densities of each subpopulation across space
        at index (time) t
        """
        fig = plt.figure(figsize=Constants.plot_size)
        sub = fig.add_subplot(111)
        self.get_subplot(sub, t)
        sub.legend(["N", "Hyb"] + Constants.subpop_legend, fontsize=8,
                   bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        fig.show()
        return fig

    def plot_size_history(self, log=True):
        """
        Make a plot of per-genotype population sizes over time
        """
        n_vec = self.generation_sizes
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        times = np.arange(self.g + 1)
        sub.plot(times, n_vec, color="black")
        for i in np.arange(9):
            sub.plot(times, np.sum(self.arr[:, :, i], axis=1),
                     color=Constants.genotype_colors[i], linewidth=2)
        sub.set_xlim(0, np.max(times))
        sub.invert_xaxis()
        if log:
            sub.set_yscale("log")
        else:
            y_lim = np.round(self.params.K * 1.1)
            sub.set_ylim(0, y_lim)
        sub.set_xlabel("t before present")
        sub.set_ylabel("population size")
        sub.legend(["N"] + Constants.subpop_legend, fontsize=8)
        fig.show()

    def plot_history(self, plot_int):
        """
        Plot several generations on a single figure to provide snapshots of
        simulation history
        """
        snaps = np.arange(self.g, -1, -plot_int)
        n_figs = len(snaps)
        if n_figs in Constants.shape_dict:
            n_rows, n_cols = Constants.shape_dict[n_figs]
        else:
            n_rows = 2
            n_cols = (n_figs + 1) // 2
        plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
        figure.tight_layout(pad=3.0)
        figure.subplots_adjust(right=0.9)
        for i in np.arange(n_figs):
            t = snaps[i]
            index = np.unravel_index(i, plot_shape)
            ax = axs[index]
            self.get_subplot(ax, t=t)
        if n_figs < plot_shape[0] * plot_shape[1]:
            index = np.unravel_index(n_figs, plot_shape)
            allele_arr = AlleleArr.from_subpop_arr(self[0])
            ax = axs[index]
            allele_arr.get_subplot(ax)
        figure.legend(["N", "Hyb"] + Constants.subpop_legend, fontsize=10,
                      loc='right', borderaxespad=0, fancybox=False,
                      framealpha=1, edgecolor="black")
        figure.show()
        return figure


class AlleleArr:

    time_axis = 0
    space_axis = 1
    locus_axis = 2
    allele_axis = 3

    def __init__(self, arr, params, t, bin_size):
        self.arr = arr
        self.params = params
        self.t = t
        self.g = params.g
        self.bin_size = bin_size

    @classmethod
    def from_generation(cls, generation_table, bin_size=0.01):
        """
        Get an AlleleArr of time dimension 1, recording the allele distribution
        in a single Generation
        """
        bins, n_bins = plot_util.get_bins(0.01)
        x = generation_table.cols.x
        alleles = generation_table.cols.alleles
        loci = np.array([[0, 1], [0, 1], [2, 3], [2, 3]])
        arr = np.zeros((1, n_bins, 2, 2), dtype=np.int32)
        for i in np.arange(4):
            j, k = np.unravel_index(i, (2, 2))
            a = i % 2 + 1
            arr[0, :, j, k] = (
                    np.histogram(x[alleles[:, loci[i, 0]] == a], bins)[0]
                    + np.histogram(x[alleles[:, loci[i, 1]] == a], bins)[0])
        params = generation_table.params
        t = generation_table.t
        return cls(arr, params, t, bin_size)

    @classmethod
    def from_pedigree(cls, pedigree_table, bin_size=0.01):
        """
        Derive an AlleleArr from an entire pedigree generation by generation

        :param pedigree_table:
        :param bin_size:
        :return:
        """
        t_dim = pedigree_table.g + 1
        n_bins = plot_util.get_n_bins(bin_size)
        arr = np.zeros((t_dim, n_bins, 2, 2), dtype=np.int32)
        for t in np.arange(t_dim):
            generation = pedigree_table.get_generation(t)
            arr[t, :, :, :] = AlleleArr.from_generation(generation).arr
        params = pedigree_table.params
        t = pedigree_table.t
        return cls(arr, params, t, bin_size)

    @classmethod
    def from_subpop_arr(cls, subpop_arr):
        """
        Convert data from a SubpopArr into an AlleleArr
        """
        manifold = Constants.allele_manifold
        arr = np.sum(subpop_arr.arr[:, :, :, None, None] * manifold, axis=2)
        params = subpop_arr.params
        t = subpop_arr.t
        bin_size = subpop_arr.bin_size
        return cls(arr, params, t, bin_size)

    def __repr__(self):
        return (f"AlleleArr of {len(self)} generations, t = {self.t}, "
                f"g = {self.g}, holding {self.n_alleles} alleles from "
                f"{self.size} organisms")

    def __str__(self):
        pass

    def __len__(self):
        """Return the number of generations represented in the array"""
        return np.shape(self.arr)[0]

    def __getitem__(self, index):
        """
        Return the generation or generations at the times or mask designated
        by index
        """
        arr = self.arr[[index]]
        params = self.params
        bin_size = self.bin_size
        return AlleleArr(arr, params, index, bin_size)

    @property
    def n_alleles(self):
        """
        Return the total number of alleles held in the array
        """
        return np.sum(self.arr)

    @property
    def size(self):
        """
        Return the total number of organisms represented in the array
        """
        return np.sum(self.arr) // 4

    @property
    def allele_densities(self):
        """
        Return an array of total allele counts per bin and time
        """
        return np.sum(self.arr, axis=3)

    @property
    def densities(self):
        """
        Return an array of organism counts per bin and time
        """
        return np.sum(self.arr, axis=(2, 3)) // 4

    @property
    def freq(self):
        """
        Return an array of allele frequencies
        """
        return self.arr / self.allele_densities[:, :, :, np.newaxis]

    @property
    def generation_freq(self):
        """
        Return the total allele frequencies for each generation
        """
        counts = np.sum(self.arr, axis=(1, 3))[:, :, None]
        return np.sum(self.arr, axis=1) / counts

    def get_size(self, t):
        """
        Return the population size of generation t
        """
        return np.sum(self.arr[t]) // 4

    def get_allele_density(self, t):
        """
        Return a vector holding the number of loci represented in each
        spatial bin at time t
        """
        return np.sum(self.arr[t, :, :, :], axis=2)

    def get_freq(self, t):
        """
        Return an array of allele frequencies at time t
        """
        n_loci = self.allele_densities
        return self.arr[t] / n_loci[t, :, :, np.newaxis]

    def get_subplot(self, sub, t=0):
        freqs = self.get_freq(t)
        bin_mids = plot_util.get_bin_mids(self.bin_size)
        for i in np.arange(3, -1, -1):
            j, k = np.unravel_index(i, (2, 2))
            sub.plot(bin_mids, freqs[:, j, k],
                     color=Constants.allele_colors[i], linewidth=2,
                     label=Constants.allele_legend[i], marker="x")
        title = "t = " + str(self.t) + " n = " + str(self.get_size(t))
        plot_util.setup_space_plot(sub, 1.01, "allele freq", title)

    def plot_freq(self, t=0):
        """
        Make a plot of the densities of each subpopulation across space
        at index (time) t
        """
        fig = plt.figure(figsize=Constants.plot_size)
        sub = fig.add_subplot(111)
        self.get_subplot(sub, t)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        fig.show()
        return fig

    def plot_freq_history(self):
        """
        Make a plot of allele frequencies over time
        """
        generation_freq = self.generation_freq
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        times = np.arange(self.g + 1)
        for i in np.arange(3, -1, -1):
            j, k = np.unravel_index(i, (2, 2))
            sub.plot(times, generation_freq[:, j, k],
                     color=Constants.allele_colors[i], linewidth=2,
                     label=Constants.allele_legend[i])
        sub.set_xlim(0, np.max(times))
        sub.invert_xaxis()
        sub.set_ylim(-0.01, 1.01)
        sub.set_xlabel("t before present")
        sub.set_ylabel("population size")
        sub.legend(fontsize=8)
        fig.show()

    def plot_history(self, plot_int):
        snaps = np.arange(self.g, -1, -plot_int)
        n_figs = len(snaps)
        if n_figs in Constants.shape_dict:
            n_rows, n_cols = Constants.shape_dict[n_figs]
        else:
            n_rows = 2
            n_cols = (n_figs + 1) // 2
        plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        figure, axs = plt.subplots(n_rows, n_cols, figsize=size, sharex='all')
        figure.tight_layout(pad=3.0)
        figure.subplots_adjust(right=0.9)
        for i in np.arange(n_figs):
            t = snaps[i]
            index = np.unravel_index(i, plot_shape)
            ax = axs[index]
            self.get_subplot(ax, t=t)
        if n_figs < plot_shape[0] * plot_shape[1]:
            index = np.unravel_index(n_figs, plot_shape)
            figure.delaxes(axs[index])
        figure.legend(Constants.allele_legend, fontsize=10, loc='right',
                      borderaxespad=0, fancybox=False, framealpha=1,
                      edgecolor="black")
        figure.show()
        return figure


class ClinePars:

    def __init__(self, x_vec, k_vec, params, bin_size):
        if len(x_vec) != len(k_vec):
            raise AttributeError("x and k vector lengths do not match")
        self.x_vec = x_vec
        self.k_vec = k_vec
        self.params = params
        self.bin_size = bin_size

    @classmethod
    def from_pedigree(cls, pedigree_table):
        allele_arr = AlleleArr.from_pedigree(pedigree_table)
        return cls.from_allele_arr(allele_arr)

    @classmethod
    def from_genotype_arr(cls, genotype_arr):
        allele_arr = AlleleArr.from_subpop_arr(genotype_arr)
        return cls.from_allele_arr(allele_arr)

    @classmethod
    def from_allele_arr(cls, allele_arr):
        allele_freq = allele_arr.get_freq()
        a2_freq = allele_freq[:, :, 0, 1]
        x = plot_util.get_bin_mids(allele_arr.bin_size)
        params = allele_arr.params
        t_dim = params.g + 1
        x_vec = np.zeros(t_dim)
        k_vec = np.zeros(t_dim)
        for t in np.arange(t_dim):
            try:
                cline_opt = cls.optimize_logistic(x, a2_freq[t])
                k_vec[t] = cline_opt[0][0]
                x_vec[t] = cline_opt[0][1]
            except:
                k_vec[t] = -1
                x_vec[t] = -1
        bin_size = allele_arr.bin_size
        return cls(x_vec, k_vec, params, bin_size)

    def __len__(self):
        return len(self.x_vec)

    @staticmethod
    def logistic_fxn(x, k, x_0):
        return 1 / (1.0 + np.exp(-k * (x - x_0)))

    @classmethod
    def optimize_logistic(cls, x, y):
        return opt.curve_fit(cls.logistic_fxn, x, y)

    def plot(self):
        length = len(self)
        t = np.arange(length)
        fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex='all')
        x_ax, k_ax = axs[0], axs[1]
        k_ax.plot(t, self.k_vec, color="black", linewidth=2)
        x_ax.plot(t, np.full(length, 0.5), color="red", linestyle="dashed")
        x_ax.plot(t, self.x_vec, color="black", linewidth=2)
        k_ax.set_ylim(0, 200)
        x_ax.set_ylim(0, 1)
        k_ax.set_xlim(0, length)
        axs[1].set_xlabel("generations before present")
        x_ax.set_ylabel("x_0")
        k_ax.set_ylabel("k")
        x_ax.set_title("cline parameter x_0")
        k_ax.set_title("cline parameter k")
        x_ax.invert_xaxis()
        fig.suptitle("Cline Parameters")
        fig.tight_layout(pad=1.0)
        fig.show()

    def plot_clines(self, n=10):
        """Plot the cline approximation at n even intervals in time"""
        snaps = np.linspace(len(self) - 1, 0, n, dtype=np.int32)
        x = plot_util.get_bin_mids(self.bin_size)
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        colors = matplotlib.cm.YlGnBu(np.linspace(0.2, 1, n))
        for i in np.arange(n):
            t = snaps[i]
            y = self.logistic_fxn(x, self.k_vec[t], self.x_vec[t])
            sub.plot(x, y, color=colors[i], linewidth=2)
        sub = plot_util.setup_space_plot(sub, 1.01, "$A^2$ cline", "Clines")
        sub.legend(snaps)
        fig.show()


class Constants:
    """
    Defines some useful objects and conventions. Importantly, all objects
    pertaining to genotypes are presented in the order
        0 : A 1, B 1
        1 : A 1, B H
        2 : A 1, B 2
        3 : A H, B 1
        4 : A H, B H
        5 : A H, B 2
        6 : A 2, B 1
        7 : A 2, B H
        8 : A 2, B 2
    """
    plot_size = (8, 6)

    n_loci = 2
    n_A_alelles = 2
    n_B_alleles = 2
    n_genotypes = 9

    # the colors associated with alleles, in the order A^1, A^2, B^1, B^2
    allele_colors = ["red",
                     "blue",
                     "lightcoral",
                     "royalblue"]

    # the colors associated with genotypes
    genotype_colors = ["red",
                       "orange",
                       "palevioletred",
                       "chartreuse",
                       "green",
                       "lightseagreen",
                       "purple",
                       "deepskyblue",
                       "blue"]

    # names of the alleles
    allele_legend = ["$A^1$",
                     "$A^2$",
                     "$B^1$",
                     "$B^2$"]

    # names of the 9 genotypes
    subpop_legend = ["A = 1 B = 1",
                     "A = 1 B = H",
                     "A = 1 B = 2",
                     "A = H B = 1",
                     "A = H B = H",
                     "A = H B = 2",
                     "A = 2 B = 1",
                     "A = 2 B = H",
                     "A = 2 B = 2"]

    # because there are 16 possible arrangements of alleles and only 9
    # genotypes, it is convenient to sum the allele values at each locus up to
    # classify organisms by genotype. This array classifies those sums
    allele_sums = np.array([[2, 2],
                            [2, 3],
                            [2, 4],
                            [3, 2],
                            [3, 3],
                            [3, 4],
                            [4, 2],
                            [4, 3],
                            [4, 4]], dtype=np.uint8)

    # there are more possible arrangements of alleles
    # these are the ones used when creating founding generations
    genotypes = np.array([[1, 1, 1, 1],
                          [1, 1, 1, 2],
                          [1, 1, 2, 2],
                          [1, 2, 1, 1],
                          [1, 2, 1, 2],
                          [1, 2, 2, 2],
                          [2, 2, 1, 1],
                          [2, 2, 1, 2],
                          [2, 2, 2, 2]], dtype=np.uint8)

    # this is an object used to convert genotype counts into allele counts.
    # each row sums to 4, as every organism has 4 alleles
    # counts: [[A^1, A^2], [B^1, B^2]]
    allele_manifold = np.array([[[2, 0], [2, 0]],
                                [[2, 0], [1, 1]],
                                [[2, 0], [0, 2]],
                                [[1, 1], [2, 0]],
                                [[1, 1], [1, 1]],
                                [[1, 1], [0, 2]],
                                [[0, 2], [2, 0]],
                                [[0, 2], [1, 1]],
                                [[0, 2], [0, 2]]], dtype=np.uint8)

    # subplot dimensions for history plots
    shape_dict = {1: (1, 1),
                  2: (1, 2),
                  3: (1, 3),
                  4: (2, 2),
                  5: (2, 3),
                  6: (2, 3),
                  8: (2, 4),
                  9: (3, 3),
                  10: (2, 5),
                  11: (3, 4),
                  12: (3, 4),
                  16: (4, 4),
                  21: (3, 7)}


# debug
if __name__ == "__main__":
    _params = parameters.Params(10_000, 10, 0.1)

    _trial = Trial(_params, plot_int=1)
    _cols = _trial.pedigree_table.cols
    gen = _trial.pedigree_table.get_generation(0)
    _genotype_arr = GenotypeArr.from_pedigree(_trial.pedigree_table)
    _allele_arr = AlleleArr.from_pedigree(_trial.pedigree_table)
