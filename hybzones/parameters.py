import json

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import os

from hybzones import util


class Params:
    """
    A class defining simulation parameters. Almost all parameters have default
    values, so non-default parameters must be explicitly declared after
    instantiation.

    Documentation at https://docs.google.com/document/d/16BmLuji9_FA6kHPHW1C6G7iGMdRR0JPiug14B4aTG8s/edit?usp=sharing
    """

    def __init__(self, g, **kwargs):
        """
        :param g: number of generations to simulate; simulation begins at
            generation g and proceeds forward in time to generation 0
        :type g: int
        """
        self.g = g
        self.subpop_n = [5_000, 0, 0, 0, 0, 0, 0, 0, 5_000]
        self.subpop_lims = [[0, 0.5], [], [], [], [], [], [], [], [0.5, 1]]
        self.K = 10_000
        self.r = 0.5
        self.g = g
        # mating parameters
        self.c = 0.1
        self.pref_model = "undesirable"
        self.beta = 0.005
        self.mating_bound = 0.02
        self.density_bound = 0.005
        self.mating_model = "gaussian"
        # dispersal parameters
        self.delta = 0.01
        self.dispersal_model = "random"
        self.edge_model = "closed"
        self.scale_factor = 2
        self.shift_factor = 2
        # fitness parameters
        self.intrinsic_fitness = False
        self.hyb_fitness = 1.0
        self.extrinsic_fitness = False
        self.female_fitness = False
        self.mu = 0.0
        self.k_1 = -20
        self.k_2 = 20
        self.mid_1 = 0.5
        self.mid_2 = 0.5
        # scripting
        self.history_type = "pedigree_table"
        self.task = None
        # genetic parameters
        self.sample_bins = None
        self.sample_sizes = None
        self.time_cutoffs = [None, None]
        self.n_windows = 10
        self.seq_length = 1e4
        self.recombination_rate = 1e-8
        self.u = 1e-8
        self.demographic_model = "one_pop"
        self.mig_rate = None
        self.rooted = True
        self_dict = self.as_dict
        for key in kwargs:
            if key in self_dict:
                setattr(self, key, kwargs[key])
            else:
                raise AttributeError(f"{key} is not a valid parameter field")

    @classmethod
    def load(cls, filename, path=None):
        """
        Instantiate a Params instance from a .json params file

        :param filename: filename in the directory specificed by path, or if
            path is not specified, in the hybzones/parameters directory
        :param path: if specified, load filename from this path
        """
        if not path:
            root = os.getcwd()
            path = root.replace("hybzones\\hybzones", "hybzones\\parameters\\")
        filename = path + filename
        file = open(filename, 'r')
        param_dict = json.load(file)
        file.close()
        params = cls(0)
        for param in param_dict:
            setattr(params, param, param_dict[param])
        return params

    def declare_genetic_params(self):
        self.sample_sizes = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
        self.sample_bins = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4],
                            [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8],
                            [0.8, 0.9], [0.9, 1.0]]

    @classmethod
    def from_dict(cls, param_dict):
        """
        Instantiate a Params instance from a dictionary of parameters

        :param param_dict: dictionary of parameters
        """
        params = cls(0)
        for field in param_dict:
            setattr(params, field, param_dict[field])
        return params

    @classmethod
    def from_string(cls, string):
        """
        Convert a string into a params instance

        :param string: a string representation of a Params instance
        """
        return cls.from_dict(eval(string))

    @classmethod
    def from_arr(cls, arr):
        """
        Convert a 1d array of characters into a params instance

        :param arr:
        :return:
        """
        string = "".join(list(arr))
        return cls.from_string(string)

    @property
    def as_string(self):
        """
        Return a string representation of the params instance
        """
        return str(vars(self))

    @property
    def as_arr(self):
        """
        Return an array of dtype U1 holding the parameter string as single
        characters
        """
        return np.array(list(self.as_string), dtype="U1")

    @property
    def as_dict(self):
        return dict(vars(self))

    def __str__(self):
        """
        Print the value of each parameter field
        """
        _dict = self.as_dict
        _out = [f"{key :.<25} {str(_dict[key])}" for key in _dict]
        out = "\n".join(_out)
        return out

    def __repr__(self):
        """
        Print a representation of the Params instance
        """
        basic = Params(self.g)
        basic_dict = dict(vars(basic))
        self_dict = dict(vars(self))
        diffs = []
        for parameter in self_dict:
            if self_dict[parameter] != basic_dict[parameter]:
                if parameter != "g":
                    diffs.append(parameter)
        out = f"params({self.g}"
        for dif in diffs:
            out += f", {dif}={self_dict[dif]}"
        return out

    def save(self, filename, path=None):
        """
        Write a dictionary representation of the params class instance in a
        .json file. By default, write to the hybzones/parameters directory
        unless a path is specified

        :param filename:
        :param path:
        """
        if not path:
            root = os.getcwd()
            path = root.replace("hybzones\\hybzones", "hybzones\\parameters\\")
        filename = path + filename
        param_dict = vars(self)
        file = open(filename, 'w')
        json.dump(param_dict, file, indent=4)
        file.close()
        print("params file written to " + filename)

    @property
    def c_matrix(self):
        """
        Get the matrix of assortation levels defined by parameters c and
        pref_model.

        :returns: 3x3 numpy array of assortative levels
        """
        if self.pref_model in c_matrix_methods:
            c_matrix = c_matrix_methods[self.pref_model](self.c)
        else:
            name = self.pref_model
            raise Exception(f"Preference model {name} is not implemented")
        return c_matrix

    def print_c(self):
        """
        Print out a table of the assortation levels defined by parameters c
        and pref_model.
        """
        printout = np.zeros((4, 4), dtype="U16")
        printout[0, :] = ["            ", "pref B = 1", "pref B = H",
                          "pref B = 2"]
        printout[1:, 0] = ["signal A = 1", "signal A = H", "signal A = 2"]
        length = len(printout[0, 1])
        c_matrix = self.c_matrix
        strings = [" " * (length - len(str(i))) + str(i) for i in
                   np.ravel(c_matrix)]
        printout[1:, 1:] = np.reshape(strings, (3, 3))
        print(printout)

    def plot_fitness(self):
        """
        Make a plot of allelic fitness reductions for A1, A2 and the
        additive fitness of the possible signal genotypes A1A1, A1A2, A2A2,
        given a Params class instance.
        """
        x = np.linspace(0, 1, 1000)
        s1 = self.mu - self.mu / (1 + np.exp(-self.k_1 * (x - self.mid_1)))
        s2 = self.mu - self.mu / (1 + np.exp(-self.k_2 * (x - self.mid_2)))
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        sub.plot(x, s1, color="red", linestyle="dashed", label="s_1")
        sub.plot(x, s2, color="blue", linestyle="dashed", label="s_2")
        sub.plot(x, 1 - 2 * s1, color="red", label="1 - 2s_1")
        sub.plot(x, 1 - s1 - s2, color="purple", label="1 - s_1 - s_2")
        sub.plot(x, 1 - 2 * s2, color="blue", label="1 - 2s_2")
        util.setup_space_plot(sub, 1.01, "relative fitness",
                              "environmental fitness")
        fig.legend(loc="lower left")
        fig.show()

    def plot_scale(self, center=0.5):
        """
        Make a plot of normal distributions with scales self.beta and self.delta
        to give a visualization of the scale of interactions in the space.
        """
        fig = plt.figure(figsize=(8,6))
        sub = fig.add_subplot(111)
        x = np.linspace(0, 1, 1000)
        beta = util.compute_pd(x - center, self.beta)
        delta = util.compute_pd(x - center, self.delta)
        beta /= np.max(beta)
        delta /= np.max(delta)
        sub.plot(x, beta, color='red', label="mating distribution")
        sub.plot(x, delta, color='orange', label="dispersal distribution")
        util.setup_space_plot(sub, 1.01, "normalized density", "scale")
        fig.legend()
        fig.show()


def get_null_c_matrix(c):
    c_matrix = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]], dtype=np.float32)
    return c_matrix


def get_intermediate_c_matrix(c):
    h = (1 + c) / 2
    c_matrix = np.array([[1, h, c],
                         [h, 1, h],
                         [c, h, 1]],  dtype=np.float32)
    return c_matrix


def get_undesirable_c_matrix(c):
    c_matrix = np.array([[1, 1, c],
                         [c, c, c],
                         [c, 1, 1]],  dtype=np.float32)
    return c_matrix


def get_semi_undesirable_c_matrix(c):
    h = (1 + c) / 2
    c_matrix = np.array([[1, 1, c],
                         [h, h, h],
                         [c, 1, 1]], dtype=np.float32)
    return c_matrix


def get_distinct_c_matrix(c):
    c_matrix = np.array([[1, c, c],
                         [c, 1, c],
                         [c, c, 1]],  dtype=np.float32)
    return c_matrix


# map the string names of preference models to the functions which implement
# them
c_matrix_methods = {"null" : get_null_c_matrix,
                    "intermediate" : get_intermediate_c_matrix,
                    "undesirable" : get_undesirable_c_matrix,
                    "semi_undesirable": get_semi_undesirable_c_matrix,
                    "distinct" : get_distinct_c_matrix}


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')
