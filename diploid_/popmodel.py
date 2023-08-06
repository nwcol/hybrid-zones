import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import time

from diploid_.parameters import Params

from diploid_ import plot_util

from diploid_ import mating_models

from diploid_ import dispersal_models

from diploid_ import fitness_models


"""
TO-DO

X-set up genotypearr
X-get plots working
X-subplots
X-set up allelearr

-make the mating model more efficient and more comprehensible

-truncation of the pedigree!!!! how
-__repr__, __str__ for pedigree and generation tables
-sort out how to handle properties between columns vs tables (direct access
or access through table property?)

-use masking instead of deletion to handle preventing hybrids from mating
-make sure that this works

-when to sort by x and id for max consistency and least use

-replace A_alleles etc with signal_alleles (lower case, descriptive)

-does setting arrays as attributes copy them?

-emergency table extension 
-params might be stored at too many levels. redundancy. think about this

-set up models
-get simulation running
-beef up columns class
-beef up tables class
-better means of creating a pedigree with specific columns in it

-improve names of genotypearr functions, theyre pretty bad
-also do this for allelearr

-__repr__ and __str__ for everyone
-zero length array printing?
-print dtypes below columns

-pedigree sampling

NOTES
-I use 'time' as a variable name to refer to the time column, t to refer to 
an integer which represents a time in history
"""


class Columns:
    """
    The core of pedigree and generation table objects, and therefore of the
    simulation.
    """

    # these are the columns essential to the function of the pedigree. they
    # are therefore mandatory to instantiate a Columns instance. time is
    # arguably not needed but its inclusion is very convenient
    _col_names = ["ID",
                  "maternal_ID",
                  "paternal_ID",
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
        types = {"ID": np.int32,
                 "maternal_ID": np.int32,
                 "paternal_ID": np.int32,
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

        self.ID = kwargs["ID"]
        self.maternal_ID = kwargs["maternal_ID"]
        self.paternal_ID = kwargs["paternal_ID"]
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
        Adapted from the __getitem__ method in the tskit basetable class.
        Return a new Columns instance holding a subset of this instance
        using 1. an integer, 2. a slice, 3. an array of integers (index), or
        4. a boolean mask

        example
        >>>cols
        Cols with 10000 filled rows of 10000 max rows in 8 columns
        >>>cols[10]
        Cols with 1 filled rows of 1 max rows in 8 columns
        >>>cols[10].ID
        array([10])
        >>>cols[10:20]
        Cols with 10 filled rows of 10 max rows in 8 columns
        >>>cols[10, 20, 40, 100, 200]
        Cols with 5 filled rows of 5 max rows in 8 columns
        >>>cols[10, 20, 40, 100, 200].ID
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

    @property
    def A_alleles(self):
        if "alleles" not in self.col_names:
            raise AttributeError("no alleles columns exist in this instance")
        return self.alleles[:, [0, 1]]

    @property
    def B_alleles(self):
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
        sums[:, 0] = np.sum(self.A_alleles, axis=1)
        sums[:, 1] = np.sum(self.B_alleles, axis=1)
        return sums

    @property
    def genotype_codes(self):
        allele_sums = self.allele_sums
        genotype_codes = np.zeros(self.filled_rows, dtype=np.uint8)
        for i in np.arange(Constants.n_genotypes):
            genotype_codes[(allele_sums[:, 0] == Constants.allele_sums[i, 0])
                & (allele_sums[:, 1] == Constants.allele_sums[i, 1])] = i
        return genotype_codes

    def apply_ID(self, i_0=0):
        """
        Add ids to the array

        :return:
        """
        self.ID += np.arange(self.max_rows) + i_0

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

    def get_subpop_index(self, **kwargs):
        """
        Return the index of organisms with character defined in **kwargs
        using the format column=character.

        'signal' and 'preference' may be given as args.

        example
        >>>cols.get_subpop_index(sex=0)
        array([0, 1, 3, ... , 9878, 9879], dtype=np.int64)

        :param kwargs:
        :return:
        """
        index = np.arange(len(self))
        for arg in kwargs:
            new = np.nonzero(getattr(self, arg) == kwargs[arg])[0]
            index = np.intersect1d(index, new)
        return index

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

    def truncate(self, new_max=None):
        """
        Return a copy of the Columns instance shortened to new_max, or to
        self.filled_rows if no new_max is provided

        :param new_max:
        :return:
        """
        if not new_max or new_max > self.max_rows:
            new_max = self.filled_rows
        return self[:new_max]

    def asdict(self):
        """
        Return a dict of columns

        :return:
        """
        col_dict = {col_name: getattr(self, col_name) for col_name in
                    self.col_names}
        return col_dict

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


class Table:

    def __init__(self, cols, params):
        self.cols = cols
        self.params = params

    def __len__(self):
        return len(self.cols)

    @property
    def nbytes(self):
        """
        Estimate the minimum number of bytes occupied in memory by the column
        arrays

        :return:
        """
        nbytes = 0
        col_dict = self.cols.asdict()
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
    def genotype_codes(self):
        return self.cols.genotype_codes

    def truncate(self, new_max=None):
        """
        Reduce the length of the table Columns instance to new_max, or if no
        new_max is provided, reduce to self.cols.filled_rows

        :param new_max:
        :return:
        """
        if not new_max:
            new_max = self.cols.filled_rows
        # this is copying, for sure
        self.cols = self.cols[:new_max]


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
                x_.append(np.random.uniform(lower, upper, n_).astype(np.float32))
        alleles = np.vstack(alleles_)
        x = np.concatenate(x_)
        flag = np.zeros(n, dtype=np.int8)
        cols = Columns(n, n, ID=ID, maternal_ID=maternal_ID,
                       paternal_ID=paternal_ID, time=time, sex=sex, x=x,
                       alleles=alleles, flag=flag)
        cols.sort_by_x()
        cols.apply_ID()
        t = params.g
        return cls(cols, params, t)

    @classmethod
    def from_cols(cls, cols, params, t):
        """
        Instantiate a generation from a bare cols object

        :param cols:
        :param params:
        :return:
        """
        return cls(cols, params, t)

    @classmethod
    def mate(cls, parent_generation_table):
        """
        Form a new generation by mating in the previous generation

        :param parent_generation_table:
        :return:
        """
        t = parent_generation_table.t - 1
        matings = mating_models.Matings(parent_generation_table)
        n = matings.n
        ID = np.zeros(n, dtype=np.int32) # do later
        maternal_ID = matings.abs_maternal_ids
        paternal_ID = matings.abs_paternal_ids
        time = cls.get_time_col(n, t)
        sex = cls.get_random_sex(n)
        x = parent_generation_table.cols.x[matings.maternal_ids]
        alleles = matings.get_zygotes(parent_generation_table)
        flag = np.full(n, 1, dtype=np.int8)
        cols = Columns(n, n, ID=ID, maternal_ID=maternal_ID,
                       paternal_ID=paternal_ID, time=time, sex=sex, x=x,
                       alleles=alleles, flag=flag)
        params = parent_generation_table.params
        return cls(cols, params, t)

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
        Return the indices of individuals with flag=1

        :return:
        """
        return self.cols.get_subpop_index(flag=1)

    def set_flags(self, idx, flag):
        """
        This is present only in the generation table because no other table
        should set flags

        :param idx:
        :param flag:
        :return:
        """
        if "flag" not in self.cols.col_names:
            raise AttributeError("no flags column exists")
        self.cols.flag[idx] = flag

    def plot(self):
        gen_arr = GenotypeArr.from_generation(self)
        fig = gen_arr.plot_density()
        return fig


class PedigreeTable(Table):

    size_factor = 1.02

    def __init__(self, cols, params, t, g):
        super().__init__(cols, params)
        self.g = g
        self.t = g

    @classmethod
    def initialize_full(cls, params, full=True):
        filled_rows = 0
        max_rows = int(params.K * (params.g + 1) * cls.size_factor)
        ID = np.zeros(max_rows, dtype=np.int32)
        time = np.full(max_rows, -1, dtype=np.int32)
        maternal_ID = np.zeros(max_rows, dtype=np.int32)
        paternal_ID = np.zeros(max_rows, dtype=np.int32)
        sex = np.zeros(max_rows, dtype=np.uint8)
        if full:
            kwargs = {"x": np.zeros(max_rows, dtype=np.float32),
                      "alleles": np.zeros((max_rows, 4), dtype=np.uint8),
                      "flag": np.full(max_rows, -10, dtype=np.int8)}
        else:
            kwargs = dict()
        cols = Columns(filled_rows, max_rows, ID=ID, maternal_ID=maternal_ID,
                       paternal_ID=paternal_ID, time=time, **kwargs)
        g = params.g
        t = params.g
        return cls(cols, params, g, t)

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
        mask = self.time == t
        cols = self.cols[mask]
        params = self.params
        return GenerationTable.from_cols(cols, params, t)


class Trial:

    def __init__(self, params, plot_int=None):
        self.run_time_0 = time.time()
        self.run_time_vec = np.zeros(params.g + 1)
        self.report_int = max(min(100, params.g // 10), 1)
        self.plot_int = plot_int
        self.figs = []

        self.complete = False
        self.g = params.g
        self.t = params.g

        self.params = params
        self.pedigree_table = PedigreeTable.initialize_full(params)
        self.run()

    def run(self):
        print("simulation initiated @ " + self.get_time_string())
        self.run_time_vec[self.params.g] = 0
        generation_table = GenerationTable.get_founding(self.params)
        if self.plot_int:
            self.initialize_figures()
            self.enter_figure(generation_table)
        while self.t > 0:
            self.pedigree_table.append_generation(generation_table)
            generation_table = self.cycle(generation_table)
        self.pedigree_table.append_generation(generation_table)
        # truncate pedigree
        if self.plot_int:
            self.set_figure_legend()
            self.figure.show()
        self.pedigree_table.truncate()
        print("simulation complete")

    def cycle(self, parent_table):
        """
        Advance the simulation through a single cycle
        """
        self.update_t()
        generation_table = GenerationTable.mate(parent_table)
        dispersal_models.disperse(generation_table)
        # generation_table.fitness(self.params)
        generation_table.cols.sort_by_x()
        generation_table.cols.apply_ID(i_0=self.pedigree_table.filled_rows)
        self.report()
        if self.plot_int:
            if self.t % self.plot_int == 0:
                self.enter_figure(generation_table)
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
            time_string = self.get_time_string()
            print(f"g{self.t : > 6} complete, runtime = {run_t : >8}"
                  + f" s, averaging {mean_t : >8} s/gen, @ {time_string :>8}")

    def initialize_figures(self):
        """
        Initialize a figure with subplots
        :return:
        """
        n_figs = self.g - np.sum(np.arange(self.g) % self.plot_int != 0) + 1
        shape_dict = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3),
                      6: (2, 3), 8: (2, 4), 9: (3, 3), 10: (2, 5), 11: (3, 4),
                      12: (3, 4), 16: (4, 4), 21: (3, 7)}
        if n_figs in shape_dict:
            n_rows, n_cols = shape_dict[n_figs]
        else:
            n_rows = 2
            n_cols = (n_figs + 1) //2
        self.plot_shape = (n_rows, n_cols)
        size = (n_cols * 4, n_rows * 3)
        self.figure, self.axs = plt.subplots(n_rows, n_cols, figsize=size,
                                             sharex='all', sharey='all')
        self.figure.tight_layout(pad=3.0)
        self.figure.subplots_adjust(right=0.9)
        self.subplot_i = 0

    def enter_figure(self, generation_table):
        """
        Enter a subplot into its appropriate subplot slot

        :return:
        """
        index = np.unravel_index(self.subplot_i, self.plot_shape)
        ax = self.axs[index]
        genotype_arr = GenotypeArr.from_generation(generation_table)
        genotype_arr.get_subplot(ax)
        self.subplot_i += 1

    def set_figure_legend(self):
        self.figure.legend(["N", "Hyb"] + Constants.subpop_legend, fontsize=10,
                           loc='right', borderaxespad=0,  fancybox=False,
                           framealpha=1, edgecolor="black")

    @staticmethod
    def get_time_string():
        return str(time.strftime("%H:%M:%S", time.localtime()))


class GenotypeArr:
    time_axis = 0
    space_axis = 1
    subpop_axis = 2

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
        """Get a SubpopArr of time dimension 1, recording a single generation
        """
        bin_edges, n_bins = plot_util.get_bins(bin_size)
        arr = np.zeros((1, n_bins, Constants.n_genotypes), dtype=np.int32)
        x = generation_table.x
        genotype_codes = generation_table.genotype_codes
        for i in np.arange(Constants.n_genotypes):
            arr[0, :, i] = np.histogram(x[genotype_codes == i],
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
        params = Params.from_string(string)
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
                f"g = {self.g}, n organisms = {self.get_size()}")

    def __str__(self):
        """
        Return a more detailed summary

        :return:
        """
        # write this
        return 0

    def __len__(self):
        """
        Return the number of generations recorded in the SubpopArr eg the
        length of the zeroth 'time' axis

        :return: length
        """
        return np.shape(self.arr)[0]

    def __getitem__(self, index):
        """
        Return the generation or generations at the times or mask designated
        by index
        """
        return self.arr[index]

    def enter_generation(self, generation):
        t = generation.t
        self.arr[t, :, :] = GenotypeArr.from_generation(generation).arr[0]

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

    def get_size(self):
        """Return the total number of organisms recorded in the array"""
        return np.sum(self.arr)

    def get_generation_sizes(self):
        """Return a vector of the whole populations of each generation"""
        return np.sum(np.sum(self.arr, axis=1), axis=1)

    def get_generation_size(self, t):
        """Return the whole-population size at generation t"""
        return np.sum(self.arr[t])

    def get_hybrid_densities(self, t):
        """Compute the sum of densities of the subpopulations with one or more
        heterozygous loci at generation t
        """
        return np.sum(self.arr[t, :, 1:8], axis=1)

    def get_bin_densities(self, t):
        """Return a vector of whole population bin densities in generation t"""
        return np.sum(self.arr[t], axis=1)

    def get_subplot(self, sub, t=0):
        """

        :param t:
        :return:
        """
        b = plot_util.get_bin_mids(self.bin_size)
        n_vec = self.get_bin_densities(t)
        sub.plot(b, n_vec, color="black", linestyle='dashed', linewidth=2)
        sub.plot(b, self.get_hybrid_densities(t), color='green',
                 linestyle='dashed', linewidth=2)
        c = Constants.genotype_colors
        for i in np.arange(9):
            sub.plot(b, self.arr[t, :, i], color=c[i], linewidth=2)
        y_max = self.params.K * 1.3 * self.bin_size
        n = str(self.get_generation_size(t))
        if len(self) == 1:
            time = self.t
        else:
            time = t
        title = "t = " + str(time) + " n = " + n
        sub = plot_util.setup_space_plot(sub, y_max, "subpop density", title)

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

    def plot_history(self, log=True):
        n_vec = self.get_generation_sizes()
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


class ClinePars:
    pass


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
                f"g = {self.g}, holding {self.get_n_alleles()} alleles from "
                f"{self.get_n_organisms()} organisms")

    def __str__(self):
        pass

    def __len__(self):
        """Return the number of generations represented in the array"""
        return np.shape(self.arr)[0]

    def get_n_alleles(self):
        """Return the total number of alleles held in the array"""
        return np.sum(self.arr)

    def get_n_organisms(self):
        """Return the total number of organisms represented in the array"""
        return np.sum(self.arr) // 4

    def get_n_at_t(self, t):
        """Return the population size of generation t"""
        return np.sum(self.arr[t]) // 4

    def get_bin_n(self):
        """Return a vector holding the number of loci represented in each
        spatial bin
        """
        return np.sum(self.arr, axis=3)

    def get_bin_n_at_t(self, t):
        """Return a vector holding the number of loci represented in each
        spatial bin at time t
        """
        return np.sum(self.arr[t, :, :, :], axis=2)

    def get_freq(self):
        """Return an array of allele frequencies"""
        n_loci = self.get_bin_n()
        return self.arr / n_loci[:, :, :, np.newaxis]

    def get_freq_at_t(self, t):
        """Return an array of allele frequencies at time t"""
        n_loci = self.get_bin_n()
        return self.arr[t] / n_loci[t, :, :, np.newaxis]

    def plot_freq(self, t=0):
        fig = plt.figure(figsize=Constants.plot_size)
        sub = fig.add_subplot(111)
        freqs = self.get_freq_at_t(t)
        bin_mids = plot_util.get_bin_mids(self.bin_size)
        for i in np.arange(3, -1, -1):
            j, k = np.unravel_index(i, (2, 2))
            sub.plot(bin_mids, freqs[:, j, k],
                     color=Constants.allele_colors[i], linewidth=2,
                     label=Constants.allele_legend[i])
        title = "t = " + str(self.t) + " n = " + str(self.get_n_at_t(t))
        sub = plot_util.setup_space_plot(sub, 1.01, "allele freq", title)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        fig.show()
        return fig


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


plt.rcParams['figure.dpi'] = 100
matplotlib.use('Qt5Agg')


params = Params(10_000, 10, 0.1)
gen = GenerationTable.get_founding(params)
cols = gen.cols
trial = Trial(params, plot_int=1)

