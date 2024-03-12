import matplotlib

import matplotlib.pyplot as plt

import msprime

import numpy as np

import networkx as nx

import os

from hybzones.pedigrees import Constants

from hybzones import parameters

from hybzones import pedigrees

from hybzones import util


def explicit_coalescent(tc, r):
    """
    Run a coalescent simulation over a table collection using the
    fixed_pedigree model

    :param tc: tree collection class instance. derived from a sampled pedigree
        table
    :param params: params instance
    :return: tree sequence
    """
    ts = msprime.sim_ancestry(initial_state=tc, model="fixed_pedigree",
                              recombination_rate=r)
    return ts


def reconstructive_coalescent(ts0, r, demography):
    """
    Root a tree sequence derived from explicit_coalescence using the dtwf
    model and a given demographic model

    :param ts0: tree sequence providing the initial state
    :param params: params instance
    :demography: demography class instance
    :return: tree sequence
    """
    ts = msprime.sim_ancestry(initial_state=ts0, demography=demography,
                              model="dtwf",
                              recombination_rate=r)
    return ts


def draw_ts(ts, node_type="ind"):
    """
    Display a tree sequence as an SVG image.
    """
    ticks = [0, 5, 10, 50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000]
    if node_type == "pop":
        node_labels = {
            node.id: f"{ts.population(node.population).metadata['name']}"
            for node in ts.nodes()
        }
        tree = ts.draw_svg(y_axis=True, node_labels=node_labels,
                           size=(1500, 500), time_scale="log_time",
                           y_ticks=ticks)
    elif node_type == "ind":
        tree = ts.draw_svg(y_axis=True, size=(1500, 500),
                           time_scale="log_time", y_ticks=ticks)
    else:
        tree = None
    with open('my.svg', 'w') as svg:
        svg.write(tree)


class SamplePedigreeTable(pedigrees.PedigreeTable):

    size_factor = 0.64

    def __init__(self, pedigree_table, sample_ids,  sample_bins, sample_sizes):
        params = pedigree_table.params
        col_names = pedigree_table.cols.col_names
        self.time_index = self.get_time_index(params)
        max_rows = self.compute_max_rows(params)
        cols = pedigrees.Columns.empty(max_rows, col_names)
        cols.filled_rows = max_rows  # filled means unfilled lol
        t = self.time_index[0]
        g = params.g
        super().__init__(cols, params, t, g)
        self.sample_bins = sample_bins
        self.sample_sizes = sample_sizes
        self.get_sample(pedigree_table, sample_ids)

    @classmethod
    def sample(cls, pedigree_table, sample_bins, sample_sizes):
        """
        Sample n individuals from each bin specified in bins from the most
        recent generation
        
        :param pedigree_table: 
        :param sample_bins: list or array specifying bin edges, of length n_bins + 1
        :param sample_sizes: integer or list of length n_bins specifying sample sizes
        """
        if type(sample_sizes) == int:
            sample_sizes = [sample_sizes for i in range(len(sample_bins))]
        sample_bins = [[sample_bins[i], sample_bins[i + 1]] 
                      for i in range(len(sample_bins) - 1)]
        sample_ids = cls.get_last_gen_ids(pedigree_table.get_generation(0),
            sample_bins, sample_sizes)
        return cls(pedigree_table, sample_ids, sample_bins, sample_sizes)

    def compute_max_rows(self, params):
        return int(self.size_factor * len(self.time_index) * params.K)

    @staticmethod
    def get_time_index(params):
        """
        Return the continuous block of time defined by the time cutoffs
        parameter, or by 0 and the g parameter if cutoffs are not defined
        """
        lower = 0
        upper = params.g
        if not lower:
            lower = 0
        if not upper:
            upper = params.g
        time_index = np.arange(lower, upper + 1)
        return time_index

    def get_sample(self, pedigree_table, sample_ids):
        """

        :param pedigree_table:
        :param sample_ids:
        :return:
        """
        sample = self.sample_last_gen(pedigree_table, sample_ids)
        self.reverse_append(sample)
        for t in self.time_index[1:]:
            self.t = t
            last_sample = sample
            parent_ids = np.unique(last_sample.cols.parents)  # unique sorts
            parent_ids = parent_ids[parent_ids > -1]  # remove -1
            sample = pedigree_table[parent_ids]
            self.reverse_append(sample)
        self.reverse_truncate()
        old_id = self.cols.id
        self.old_id = np.copy(old_id)
        new_id = np.arange(len(self.cols), dtype=np.int32)
        for col_name in ["maternal_id", "paternal_id"]:
            column = getattr(self.cols, col_name)
            mask = column > -1  # we must filter out -1 values
            column[mask] = self.remap_ids(column[mask], old_id, new_id)
        self.cols.id[:] = new_id

    def sample_last_gen(self, pedigree_table, sample_ids):
        """
        Sample n organisms in n_sample_bins bins from the lowest (first)
        time defined in self.t_slice and sort them by id

        :param pedigree_table: pedigree_table
        :param sample_ids: array of last gen relative ids to sample
        """
        last_gen = pedigree_table.get_generation(self.time_index[0])
        sample = last_gen[sample_ids]
        sample = sample[sample.cols.id.argsort()]  # sort by ID
        return sample

    @staticmethod
    def get_last_gen_ids(last_gen, sample_bins, sample_sizes):
        """
        Get the relative ids to sample in the generation

        :param last_gen:
        :return:
        """
        sample_ids = []
        for sample_bin, size in zip(sample_bins, sample_sizes):
            ids = last_gen.cols.get_subpop_index(x=sample_bin)
            try:
                _sample_ids = np.random.choice(ids, size=size, replace=False)
                sample_ids.append(_sample_ids)
            except:
                pass
        sample_ids = np.concatenate(sample_ids)
        return sample_ids

    def reverse_append(self, generation_table):
        """
        Add a generation starting at the bottom of the columns

        :param generation_table:
        :return:
        """
        now_filled = self.filled_rows - len(generation_table)
        if now_filled < 0:
            raise ValueError("the pedigree table is full!")
        self.cols[now_filled:self.filled_rows] = generation_table.cols
        self.cols.filled_rows = now_filled

    def reverse_truncate(self, new_min=None):
        if not new_min:
            new_min = self.filled_rows
        self.cols.reverse_truncate(new_min)

    @staticmethod
    def remap_ids(id_col, old_id, new_id):
        """
        Remap the IDs in ID_col using the map old_ID[i] -> new_ID[i], eg
        from a column of increasing integers with gaps to one without gaps

        :param id_col: the column undergoing mapping
        :param old_id: the domain of the map
        :param new_id: the codomain of the map
        :return:
        """
        index = np.searchsorted(old_id, id_col)
        return new_id[index]

    def get_generation_sizes(self):
        """
        Return the number of individuals in each generation of the sample
        """
        length = len(self.time_index)
        sizes = np.zeros(length, dtype=np.int64)
        for i, t in zip(np.arange(length), self.time_index):
            sizes[i] = self.cols.get_subpop_size(time=t)
        return sizes

    def get_tc(self, demography, demographic_model, seq_length):
        """
        Build a pedigree table collection from the sample pedigree array
        """
        ped_tc = msprime.PedigreeBuilder(demography=demography)
        #  INDS
        n = len(self)
        ind_flag = self.cols.flag.astype(np.uint32)
        parents = np.ravel(self.cols.parents)
        parents_offset = np.arange(n + 1, dtype=np.uint32) * 2
        ped_tc.individuals.append_columns(flags=ind_flag, parents=parents,
                                          parents_offset=parents_offset)
        # NODES
        node_flag = np.repeat(ind_flag, 2)
        node_time = np.repeat(self.cols.time, 2).astype(np.float64)
        pop_methods = {"three_pop": self.get_three_pop_ids,
                       "one_pop": self.get_one_pop_ids}
        ind_pop = pop_methods[demographic_model]()
        node_pop = np.repeat(ind_pop, 2)
        node_ind = np.repeat(np.arange(n, dtype=np.int32), 2)
        ped_tc.nodes.append_columns(flags=node_flag, time=node_time,
                                    population=node_pop, individual=node_ind)
        ped_tc = ped_tc.finalise(sequence_length=seq_length)
        return ped_tc

    def get_one_pop_demography(self):
        """
        Construct a basic demography with a single population, pop0

        :param params: parameter set
        :return: demography instance
        """
        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=self.params.K)
        return demography

    def get_three_pop_demography(self, mig_rate):
        """
        Construct a basic demography with three populations

        All organisms after the founding generation are pop0; founders of
        population 1 are pop1, population 2 are pop2.

        A symmetric migration rate exists between ancestral populations pop1
        and pop2 to allow coalescence to occur

        :param params: parameter set
        :return: demography instance
        """
        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=self.params.K)
        demography.add_population(name="pop1", initial_size=self.params.K // 2)
        demography.add_population(name="pop2", initial_size=self.params.K // 2)
        demography.set_symmetric_migration_rate(["pop1", "pop2"], mig_rate)
        return demography

    def get_three_pop_ids(self):
        """
        Get population names for each individual for the 3-population
        demographic model
        """
        n = len(self)
        ind_pops = np.zeros(n, dtype=np.int32)
        max_t = np.max(self.time_index)
        pop1_index = self.cols.get_subpop_index(time=max_t, genotype=0)
        pop2_index = self.cols.get_subpop_index(time=max_t, genotype=8)
        ind_pops[pop1_index] = 1
        ind_pops[pop2_index] = 2
        return ind_pops

    def get_one_pop_ids(self):
        """
        Get population names for each individual for the 1-population
        demographic model
        """
        n = len(self)
        ind_pops = np.zeros(n, dtype=np.int32)
        return ind_pops

    def draw(self, x_lim=None):
        """
        Draw a pedigree tree collection as a network using networkx
        """
        G = nx.DiGraph()
        F = nx.DiGraph()
        M = nx.DiGraph()
        genotype = self.cols.genotype
        parents = self.cols.parents
        for id in self.cols.id:
            time = self.cols.time[id]
            geno = genotype[id]
            G.add_node(id, time=time, genotype=geno)
            for i in [0, 1]:
                parent = parents[id, i]
                if parent != -1:
                    G.add_edge(id, parent)
            if self.cols.sex[id] == 0:
                F.add_node(id, time=time, genotype=geno)
            else:
                M.add_node(id, time=time, genotype=geno)
        x = self.cols.x
        time = self.cols.time
        pos = nx.multipartite_layout(G, subset_key="time", align="horizontal",
                                     center=[x[-1], 0])
        for i in np.arange(len(pos)):
            pos[i] = np.array([x[i], time[i]])
        f_colors = [Constants.genotype_colors[node_attr["genotype"]]
                    for node_attr in F.nodes.values()]
        m_colors = [Constants.genotype_colors[node_attr["genotype"]]
                    for node_attr in M.nodes.values()]
        fig, ax = plt.subplots(figsize=(10, 6))
        nx.draw_networkx_edges(G, pos, node_size=40, width=0.5, arrows=False)
        nx.draw_networkx_nodes(F, pos, edgecolors="black", node_color=f_colors,
                               node_size=40, ax=ax)
        # nx.draw_networkx_labels(F, pos, font_size = 7)
        nx.draw_networkx_nodes(M, pos, edgecolors="black", node_color=m_colors,
                               node_shape='s', node_size=40, ax=ax)
        # nx.draw_networkx_labels(M, pos, font_size = 7)
        ax.tick_params(left=True, bottom=True, labelleft=True,
                       labelbottom=True)
        if x_lim:
            ax.set_xlim(x_lim[0], x_lim[1])
        else:
            ax.set_xlim(0,1)
        fig.show()


class SampleSet:
    """
    Class to keep track of node/individual ids in sample pedigrees and in
    tskit tree structures.
    """

    def __init__(self, ind_ids, node_ids):
        self.ind_ids = ind_ids
        self.node_ids = node_ids
        self.n_inds = np.size(ind_ids)

    def __repr__(self):
        # abuse of __repr__ but useful
        return f"SampleSet of {self.n_inds} ind ids"

    def __len__(self):
        return self.n_inds

    @classmethod
    def from_node_ids(cls, node_ids):
        ind_ids = cls.map_node_to_ind(node_ids)
        return cls(ind_ids, node_ids)

    @classmethod
    def from_ind_ids(cls, ind_ids):
        node_ids = cls.map_ind_to_node(ind_ids)
        return cls(ind_ids, node_ids)

    @staticmethod
    def map_node_to_ind(node_ids):
        """map a vector of table collection 'node' indices to table collection
        'individual' indices
        """
        ind_ids = np.array(list(set(node_ids // 2)))
        return ind_ids

    @staticmethod
    def map_ind_to_node(ind_ids):
        """transform a vector of indices of a table collection 'individuals'
        table into indices of a table collection 'nodes' table.
        """
        node_ids = np.repeat(ind_ids, 2, axis=0) * 2
        increment = np.ravel(np.repeat([[0, 1]], np.size(ind_ids), axis=0))
        node_ids += increment
        return node_ids


class MultiWindow:
    """
    Class to structure multi-window coalescence simulations over a sample
    pedigree table. There are two somewhat divergent functions for the class;
    it functions to simulate coalescence and gather statistic as well as to
    load existing statistics created by instances of itself and analyze them
    """

    def __init__(self, sample_pedigree, n_windows, seq_length=1e4, r=1e-8, 
                 u=1e-8, rooted=True, demog="one_pop", mig_rate=None):
        self.sample_bins = sample_pedigree.sample_bins
        self.sample_sizes = sample_pedigree.sample_sizes
        self.n_windows = n_windows
        self.seq_length = seq_length
        self.r = r
        self.u = u
        self.rooted = rooted
        self.demographic_model = demog
        self.mig_rate = mig_rate
        n_bins = len(self.sample_bins)
        self.pi = np.zeros((n_windows, n_bins))
        self.genotype_pi = np.zeros((n_windows, 9, n_bins))
        self.pi_xy = np.zeros((n_windows, n_bins, n_bins))
        if demog == "one_pop":
            self.demography = sample_pedigree.get_one_pop_demography()
        elif demog == "three_pop":
            self.demography = sample_pedigree.get_three_pop_demography(
                mig_rate)
        self.sample_bins = sample_pedigree.sample_bins
        self.sample_sets = self.get_sample_sets(sample_pedigree)
        self.genotype_sample_sets = self.get_genotype_sample_sets(
            sample_pedigree)
        tc = sample_pedigree.get_tc(self.demography, demog, seq_length)
        self.run(tc)
        self.window_kb = self.seq_length / 1000
        self.total_Mb = self.window_kb * self.n_windows / 1000

    @classmethod
    def deprecated(cls, sample_pedigree_table):
        """
        Run simulations over a sample pedigree table as defined in the table's
        params instance.

        :param sample_pedigree_table:
        :return:
        """
        params = sample_pedigree_table.params
        n_bins = len(params.sample_bins)
        pi = np.zeros((params.n_windows, n_bins))
        genotype_pi = np.zeros((params.n_windows, 9, n_bins))
        pi_xy = np.zeros((params.n_windows, n_bins, n_bins))
        inst = cls(pi, pi_xy, genotype_pi, params)
        inst.demography = sample_pedigree_table.demography
        inst.sample_bins = sample_pedigree_table.sample_bins
        inst.sample_sets = inst.get_sample_sets(sample_pedigree_table)
        inst.genotype_sample_sets = inst.get_genotype_sample_sets(
            sample_pedigree_table)
        print(f"sampling {inst.params.n_windows} x {inst.window_kb} " 
              f"kb regions for {inst.total_Mb} Mb total")
        tc = sample_pedigree_table.get_tc()
        inst.run(tc)
        return inst

    def run(self, tc):
        for i in np.arange(self.n_windows):
            pi, pi_xy, genotype_pi = self.get_window(tc)
            self.pi[i, :] = pi
            self.pi_xy[i, :, :] = pi_xy
            self.genotype_pi[i, :, :] = genotype_pi
            print(f"window {i} complete @ " + util.get_time_string())
        print("Multi-window sampling complete @ " + util.get_time_string())

    def get_sample_sets(self, sample_pedigree_table):
        """
        Get a list of SampleSet objects. Each SampleSet instance holds the
        individual and node IDs of the sample individuals in a spatial bin

        :param sample_pedigree_table:
        :return: sample_sets
        """
        sample_sets = []
        last_gen = sample_pedigree_table.get_generation(0)
        full_gen_ids = last_gen.cols.id  # all absolute ids
        for sample_bin in self.sample_bins:
            relative_ids = last_gen.cols.get_subpop_index(x=sample_bin)
            abs_ids = full_gen_ids[relative_ids]
            sample_sets.append(SampleSet.from_ind_ids(abs_ids))
        return sample_sets

    def get_genotype_sample_sets(self, sample_pedigree_table):
        """
        Return a list of SampleSet instances. Individuals are segregated into
        instances by x bin and by genotype.

        :param sample_pedigree_table:
        :return: genotype_sample_sets, a list of lists of SampleSets. sub-lists
            each correspond to a genotype and hold sample sets for spatial bins
        """
        genotype_sample_sets = []
        last_gen = sample_pedigree_table.get_generation(0)
        full_gen_ids = last_gen.cols.id
        for geno in np.arange(Constants.n_genotypes):
            genotype_sets = []
            for sample_bin in self.sample_bins:
                relative_ids = last_gen.cols.get_subpop_index(x=sample_bin,
                                                              genotype=geno)
                abs_ids = full_gen_ids[relative_ids]
                genotype_sets.append(SampleSet.from_ind_ids(abs_ids))
            genotype_sample_sets.append(genotype_sets)
        return genotype_sample_sets

    def get_window(self, tc):
        """
        Perform explicit and reconstructive coalescence simulations and
        compute diversity over spatial sample sets

        :param tc: table collection to simulate over
        """
        ts = explicit_coalescent(tc, self.r)
        if self.rooted:
            ts = reconstructive_coalescent(ts, self.r, self.demography)
        pi = self.get_diversities(ts, self.sample_sets)
        pi_xy = self.get_divergences(ts, self.sample_sets)
        n_bins = len(self.sample_bins)
        genotype_pi = np.zeros((Constants.n_genotypes, n_bins))
        for genotype in np.arange(Constants.n_genotypes):
            genotype_set = self.genotype_sample_sets[genotype]
            lengths = np.array([len(s) for s in genotype_set])
            idx = np.where(lengths > 0)[0]
            nonzero_set = []
            for i in idx:
                nonzero_set.append(genotype_set[i])
            genotype_pi[genotype, idx] = self.get_diversities(ts, nonzero_set)
        return pi, pi_xy, genotype_pi

    def get_diversities(self, ts, sample_sets):
        n_sets = len(sample_sets)
        pi = np.zeros(n_sets, dtype=np.float32)
        for i in np.arange(n_sets):
            pi[i] = ts.diversity(sample_sets=sample_sets[i].node_ids,
                                 mode="branch")
        pi *= self.u
        return pi

    def get_divergences(self, ts, sample_sets):
        n_sets = len(sample_sets)
        pi_xy = np.zeros((n_sets, n_sets), dtype=np.float32)
        for linear_idx in np.arange(np.square(n_sets)):
            idx = np.unravel_index(linear_idx, (n_sets, n_sets))
            pi_xy[idx] = ts.divergence(
                sample_sets=[sample_sets[idx[0]].node_ids,
                             sample_sets[idx[1]].node_ids],
                mode="branch")
        pi_xy *= self.u
        return pi_xy

    @property
    def mean_pi(self):
        return np.mean(self.pi, axis=0)

    @property
    def mean_pi_xy(self):
        return np.mean(self.pi_xy, axis=0)

    @property
    def mean_genotype_pi(self):
        return np.mean(self.genotype_pi, axis=0)

    def save(self, prefix, suffix):
        """
        Save each diversity/divergence array which has been created as a .txt
        file

        :param prefix: the parameter file name
        :type prefix: string
        :param suffix: the cluster and process id
        :type suffix: string
        """
        filename = prefix + "_multiwindow_" + suffix
        param_arr = self.params.as_arr
        np.savez(filename, pi=self.pi, pi_xy=self.pi_xy,
                 genotype_pi=self.genotype_pi, param_arr=param_arr)

    def plot_pi(self, ylim=None, title=None):
        """
        Plot the mean of each multi-window pi arr in self.pi_list
        """
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        x = np.round(np.arange(0, 1, 1 / len(self.sample_bins)), decimals=3)
        mean = np.mean(self.pi, axis=0)
        std = np.std(self.pi, axis=0)
        sub.errorbar(x, mean, yerr=std, capsize=2, color="black")
        sub.set_xlim(-0.1, 1)
        if ylim:
            sub.set_ylim(ylim)
        sub.set_xticks(np.arange(0, 1.1, 0.1))
        sub.grid(visible=True)
        sub.set_ylabel("nucleotide diversity")
        sub.set_xlabel("spatial bin (left edge)")
        if title:
            sub.set_title(title)

    def plot_genotype_pi(self, ylim=None):
        """
        Plot the mean genotype pis for each multi-window
        """
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        x = np.round(np.arange(0, 1, 1 / len(self.sample_bins)), decimals=3)
        for j in np.arange(Constants.n_genotypes):
            sub.plot(x, self.mean_genotype_pi[j], linewidth=2,
                color=Constants.genotype_colors[j])
        sub.set_xticks(np.arange(0, 1.1, 0.1))
        sub.set_xlim(-0.1, 1)
        sub.grid(visible=True)
        sub.set_ylabel("nucleotide diversity")
        sub.set_xlabel("spatial bin (left edge)")
        if ylim:
            sub.set_ylim(ylim)

    def plot_pi_xy(self, title=None, vmin=None, vmax=None):
        """
        Plot the mean of a set of divergence arrays using a heatmap
        """
        z = self.mean_pi_xy
        shape = np.shape(z)
        _y = _x = np.round(np.arange(0, 1, 1 / shape[0]), decimals=3)
        x, y = np.meshgrid(_x, _y)
        fig, ax = plt.subplots(figsize=(7.5, 6))
        if not vmax:
            vmax = np.max(z)
        if not vmin:
            vmin = np.min(z)
        colormesh = ax.pcolormesh(x, y, z, cmap="plasma", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(colormesh)
        plt.xlabel("x coordinate bin")
        plt.ylabel("x coordinate bin")
        if title:
            ax.set_title(title)


class MultiWindowCollection:
    """
    Class holding many instances of MultiWindow
    """

    def __init__(self, multi_window_list):
        self.params = multi_window_list[0].params
        self.n = len(multi_window_list)
        self.n_bins = np.shape(multi_window_list[0].pi)[1]
        # objects holding all data
        self.pi_list = []
        self.pi_xy_list = []
        self.genotype_pi_list = []
        for multi_window in multi_window_list:
            self.pi_list.append(multi_window.pi)
            self.pi_xy_list.append(multi_window.pi_xy)
            self.genotype_pi_list.append(multi_window.genotype_pi)
        self.pi_arr = np.stack(self.pi_list, axis=0)
        self.pi_xy_arr = np.stack(self.pi_xy_list, axis=0)
        self.genotype_pi_arr = np.stack(self.genotype_pi_list, axis=0)

    @classmethod
    def load_dir(cls, directory):
        """
        Directory is assumed to be in hybzones/data
        """
        base_dir = os.getcwd().replace(r"\hybzones", "") + r"\hybzones"
        directory = base_dir + "\\data\\" + directory + "\\"
        file_names = os.listdir(directory)
        file_names = [directory + filename for filename in file_names]
        multi_window_list = []
        for file_name in file_names:
            multi_window_list.append(MultiWindow.load(file_name))
        return cls(multi_window_list)

    @property
    def mean_trial_pi(self):
        """
        Get the mean pi within each trial
        """
        return np.mean(self.pi_arr, axis=1)

    @property
    def mean_trial_pi_xy(self):
        """
        Get the mean pi_xy within each trial
        """
        return np.mean(self.pi_xy_arr, axis=1)

    @property
    def mean_trial_genotype_pi(self):
        """
        Get the genotype pis within each trial
        """
        return np.mean(self.genotype_pi_arr, axis=1)

    @property
    def mean_pi(self):
        """
        The mean pi between all trials
        """
        return np.mean(self.mean_trial_pi, axis=0)

    @property
    def mean_pi_xy(self):
        return np.mean(self.mean_trial_pi_xy, axis=0)

    def plot_pi(self, title=None, ylim=None):
        """
        Plot an array of pi vectors using violin plots and error bars for the
        mean diversities.
        """
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        mean_pi = self.mean_trial_pi
        n = np.shape(mean_pi)[0]
        bin_width = 1 / self.n_bins
        x = np.round(np.arange(0, 1, 1 / self.n_bins), decimals=3)
        parts = sub.violinplot(mean_pi, positions=x, widths=bin_width / 2,
                               showextrema=False)
        sub.scatter(x, np.median(mean_pi, axis=0), color="black",
                    label="medians", marker='^')
        means = np.mean(mean_pi, axis=0)
        sub.errorbar(x, means, yerr=np.std(mean_pi, axis=0), color="black",
                     label="means", capsize=2)
        for pc in parts['bodies']:
            pc.set_facecolor('white')
            pc.set_edgecolor('black')
            pc.set_alpha(0.3)
        sub.set_xlim(-0.1, 1)
        if ylim:
            sub.set_ylim(ylim)
        else:
            sub.set_ylim(np.min(mean_pi) * 0.98, np.max(mean_pi) * 1.02)
        sub.legend(loc='upper right')
        sub.set_xticks(np.arange(0, 1.1, 0.1))
        sub.grid(visible=True)
        sub.set_ylabel("nucleotide diversity")
        sub.set_xlabel("spatial bin (left edge)")
        if title:
            sub.set_title(title + ", n = " + str(n))
        fig.show()

    def plot_mean_pi(self, ylim=None, title=None):
        """
        Plot the mean of each multi-window pi arr in self.pi_list
        """
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        x = np.round(np.arange(0, 1, 1 / self.n_bins), decimals=3)
        for i in np.arange(len(self.pi_list)):
            means = np.mean(self.pi_list[i], axis=0)
            stds = np.std(self.pi_list[i], axis=0)
            sub.errorbar(x, means, yerr=stds, capsize=2, color="black")
        sub.set_xlim(-0.1, 1)
        if ylim:
            sub.set_ylim(ylim)
        sub.set_xticks(np.arange(0, 1.1, 0.1))
        sub.grid(visible=True)
        sub.set_ylabel("nucleotide diversity")
        sub.set_xlabel("spatial bin (left edge)")
        if title:
            sub.set_title(title)

    def plot_mean_genotype_pi(self, ylim=None):
        """
        Plot the mean genotype pis for each multi-window
        """
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        x = np.round(np.arange(0, 1, 1 / self.n_bins), decimals=3)
        for i in np.arange(self.n):
            genotype_pi = self.mean_trial_genotype_pi[i]
            for j in np.arange(Constants.n_genotypes):
                sub.plot(x, genotype_pi[j], linewidth=2,
                         color=Constants.genotype_colors[j])
        sub.set_xticks(np.arange(0, 1.1, 0.1))
        sub.set_xlim(-1 / self.n_bins, 1)
        sub.grid(visible=True)
        sub.set_ylabel("nucleotide diversity")
        sub.set_xlabel("spatial bin (left edge)")
        if ylim:
            sub.set_ylim(ylim)

    def plot_genotype_pi(self, ylim=None):
        fig = plt.figure(figsize=(8, 6))
        sub = fig.add_subplot(111)
        x = np.round(np.arange(0, 1, 1 / self.n_bins), decimals=3)
        for i in np.arange(Constants.n_genotypes):
            genotype_pis = self.mean_trial_genotype_pi[:, i]
            parts = sub.violinplot(genotype_pis, positions=x, widths=0.05,
                                   showextrema=False)
            sub.scatter(x, np.median(genotype_pis, axis=0),
                        color=Constants.genotype_colors[i])
            for pc in parts['bodies']:
                pc.set_facecolor(Constants.genotype_colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(0.3)
        sub.set_xlim(-1 / self.n_bins, 1)
        sub.grid(visible=True)
        sub.set_ylabel("nucleotide diversity")
        sub.set_xlabel("spatial bin (left edge)")
        if ylim:
            sub.set_ylim(ylim)

    def plot_pi_xy(self, title=None, vmin=None, vmax=None):
        """
        Plot the mean of a set of divergence arrays using a heatmap
        """
        z = self.mean_pi_xy
        shape = np.shape(z)
        _y = _x = np.round(np.arange(0, 1, 1 / shape[0]), decimals=3)
        x, y = np.meshgrid(_x, _y)
        fig, ax = plt.subplots(figsize=(7.5, 6))
        if not vmax:
            vmax = np.max(z)
        if not vmin:
            vmin = np.min(z)
        colormesh = ax.pcolormesh(x, y, z, cmap="plasma", vmin=0.00018, vmax=0.00038)
        cbar = plt.colorbar(colormesh)
        plt.xlabel("x coordinate bin")
        plt.ylabel("x coordinate bin")
        if title:
            ax.set_title(title)

    def get_summary_stats(self):
        """
        Pull a few summary statistics deemed useful for comparing trials
        """
        center = self.n_bins // 2
        center_pi = np.mean(self.mean_pi[center-1:center+1])
        edge_pi = np.mean([self.mean_pi[0], self.mean_pi[-1]])
        edge_pi_xy = np.mean([self.mean_pi_xy[0, -1], self.mean_pi_xy[-1, 0]])
        # should be the same anyway
        return {"rooted": self.params.rooted,
                "g": self.params.g,
                "center_pi": center_pi,
                "edge_pi": edge_pi,
                "edge_pi_xy": edge_pi_xy,
                "n": self.n}


def plot_summary_stats(dict_list):
    fig = plt.figure(figsize=(8, 6))
    sub = fig.add_subplot(111)
    rooted = np.array([group["rooted"] for group in dict_list])
    g = np.array([group["g"] for group in dict_list])
    center_pi = np.array([group["center_pi"] for group in dict_list])
    edge_pi = np.array([group["edge_pi"] for group in dict_list])
    edge_pi_xy = np.array([group["edge_pi_xy"] for group in dict_list])
    n = np.array([group["n"] for group in dict_list])
    for val in [True, False]:
        if val:
            color = "blue"
        else:
            color = "red"
        mask = np.array(rooted) == val
        sub.plot(g[mask], center_pi[mask], marker='v', color=color,
                 label=f"center bin pi, rooted: {val}")
        sub.plot(g[mask], edge_pi[mask], marker='o', color=color,
                 label=f"edge bin pi, rooted: {val}")
        sub.plot(g[mask], edge_pi_xy[mask], marker='s', color=color,
                 label=f"edge pi_xy, rooted: {val}")
        for i, label in enumerate(n[mask]):
            sub.annotate(label, (g[mask][i], edge_pi_xy[mask][i]))
    sub.plot(g[rooted == True], center_pi[rooted] - center_pi[rooted==False],
             color="black", label="center pi diff", marker='v')
    sub.plot(g[rooted == True], edge_pi[rooted] - edge_pi[rooted==False],
             color="black", label="edge pi diff", marker='o')
    sub.plot(g[rooted == True], edge_pi_xy[rooted] - edge_pi_xy[rooted==False],
             color="black", label="edge pi_xy diff", marker='s')
    sub.legend()
    sub.set_xlim(0, 25_000)
    sub.set_xlabel("g. of explicit simulation")
    sub.set_ylim(0, 0.0005)
    sub.set_ylabel("nucleotide diversity/divergence")
    fig.show()
