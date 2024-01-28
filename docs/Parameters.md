## Introduction
The `Params` class is a required argument to instantiate the `Trial` class. Its role is to store an immutable collection of simulation parameters. During simulation, the `Params` instance is repeatedly passed to lower-level objects and made an attribute of structures such as the pedigree table, where its attributes are accessed to govern a variety of processes. Params instances can be saved as JSON format files. 

The only required argument to initialize a Params instance is `g`, the number of generations to simulate; all other attributes have default values which can be overridden by keyword arguments.


## Parameters

Here is a list of Params attributes and their types, biological interpretations, and functions in the simulation. 

`subpop_n` : list of int, default `[N/2, 0, 0, 0, 0, 0, 0, 0, N/2]`

Defines the initial population sizes of each subpopulation, i.e. genotype. By default, 5000 individuals of each genotype A¹A¹B¹B¹, A²A²B²B² will compose the founding population.

`subpop_lims` : list of lists of int

Defines the limits in which each initial subpopulation exists. Individual positions are drawn uniformly at random from the defined intervals

`K` : int, default `N`

Defines the carrying capacity of the entire space. Carrying capacity is assumed to be uniform, such that a subset of the space (x₀, x₁) is understood to have a carrying capacity of K(x₁ - x₀). 

`r` : float, default 0.5

Defines the population growth rate by setting the maximum expected family size. Expected family sizes are density-dependent, 

`g` : int

Defines the duration of the simulation in generations. Simulations endure for this exactly defined time. 

`c` : float

Defines the strength of pure assortative mating. Specifically, c sets the ratio between the chance that, all other things being equal, a homozygous-preference female mates with an iso-signal homozygous-signal male and an allo-signal homozygous-signal male. The other assortative strengths are determined as a function of c by the preference model.

`pref_model` : string

Defines the preference model, which is a function from c to a 3x3 matrix which gives the relative chances (all other things being equal) of all genotypic mating pairings.

`beta` : float, default 0.005

Defines the standard deviation of the distribution from which the mating probability masses are sampled. 

`bound` : float, default 0.02

Defines an upper limit on “interaction” in the form of mating, and on males’ perception of the signal phenotypes of other males. 

`density_bound` : float, default 0.005

Defines the extent of space within which individuals affect each other’s observations of density. These observations control reproductive output, which is density-dependent.

`mating_model` : string

Remove me lol

`delta` : float

Defines the standard deviation of the normal distribution from which individual dispersal is sampled for all individuals in the random dispersal model, and for subsets of individuals in other dispersal models.

`dispersal_model` : string, default `“random”`

Defines the dispersal model. There are three dispersal models: random, shift and scale. The random dispersal model gives every individual a dispersal sampled randomly from a normal distribution with standard deviation delta. 

The shift model samples from the same distribution for females and males with heterozygous signal genotype, but shifts the mean of the distribution for homozygous-signal males according to those males’ “perceptions” of the signal compositions to their left and right; the mean is shifted towards the direction with a higher proportion of iso-signal males, scaled proportional to the difference.

The scale model similarly treats homozygous-signal males differently, altering the standard deviations of their dispersal distributions. Their standard deviations are a function of the proportion of iso-signal males they observe about themselves; males with lower proportions will have higher standard deviations and will tend to disperse farther away.
Both nonrandom dispersal models propose a mechanism for males with lower mating chances to disperse into areas with potentially better mating prospects.

`edge_model` : string, default “closed”

`scale_factor` : float, default 2

Defines the effect strength of the “scale” dispersal model. The scale dispersal model makes the standard deviation of each homozygous-signal males’ dispersal a function of s, where

	s = frequency of iso-signal males in neighborhood

The scale function is linear and passes through the points

	(s = 0, f(s) = scale_factor * delta), ( s= 1, f(s) = 1)

So that males with no iso-signal males in their neighborhood sample their dispersal with a standard deviation scale_factor times higher than those with entirely iso-signal males in their neighborhood. The scale function is defined as 

	f(s) = ((1 - scale_factor) * s + scale_factor) * delta

`shift_factor` : float, default 2

Defines the effect strength for the “shift” dispersal model, which makes the mean of dispersal distance a function of the difference between iso-signal frequency in the neighborhoods to the left and right of homozygous-signal males. Concretely,

	s_r = frequency of iso-signal males to right 
	s_l = frequency of iso-signal males to left

	f(s) = (s_r - s_l) * shift_factor * delta

`intrinsic_fitness` : bool, default False

If True, exert an intrinsic fitness effect on heterozygote-signal individuals. 

`hyb_fitness` = float, default 1.0

If intrinsic_fitness is set to True, hyb_fitness gives the relative “innate” fitness of signal heterozygotes. Intrinsic fitness is so named because if True, it operates independently of spatial position as a uniform reduction in hybrid success. Whether intrinsic fitness affects females is governed by the same’ 

`extrinsic_fitness` : bool, default `False`

Defines whether environmental e.g. spatially variable fitness should operate on the population. Extrinsic fitness acts additively on the signal alleles. Fitness is simulated as a probability of survival, with a relative fitness of 1.0 guaranteeing survival. Reductions in fitness per allele are computed as logistic functions of spatial position x, parameterized using the k_1, k_2, mid_1, and mid_2 parameters. Here are the fitness functions:

	P(survival | a_0 = i, a_1 = j, x_pos = x) = 1 - S_i(x) - S_j(x)

where

	S_1(x) = mu - mu / (1 + exp{-k_1(x - mid_1)})
	S_2(x) = mu - mu / (1 + exp{-k_2(x - mid_2)})


`female_fitness`: bool, default `False`

Defines whether females are affected by intrinsic and extrinsic fitness. 

`k_1` : float, default -20

Defines the slope of the fitness curve for A¹.	

`k_2` : float, default 20

Defines the slope of the fitness curve for A². 

`mu` : float, default 0

Defines the maximum reduction in fitness per allele. For a homozygous-signal individual, fitness at the opposite edge of space equals approximately 1 - mu * 2.

`mid_1` : float, default 0.5

Defines the midpoint/inflection point of the fitness curve for A¹.

`mid_2` : float, default 0.5

Defines the midpoint/inflection point of the fitness curve for A². 

`history_type` : string, default “pedigree_table”

Defines the simulation output type. There are two options: pedigree_table and genotype_arr. Genotype arrays act as a summary of population history and are substantially smaller objects than pedigree tables. They are therefore useful for studying the effects of parameters when the detail provided by a pedigree table is not necessary. Specifically, a genotype array is a 3-dimensional numpy array where genotype_arr.arr[t] contains 9 histograms recording the density of each subpopulation at time t.

`task` : string, default “get_multi_window”

`sample_sizes` : list of ints

`sample_bins` : list of list of floats

`time_cutoffs` : list of ints

remove?

`n_windows` : int

The number of genome windows to simulate coalescence over. For instance, with n_windows = 1000 and seq_length = 10000, effectively 10 Mb of the genome will be simulated.

`mig_rate` : float, default 1e-4

If demographic_model == “three_pop”, defines the migration rate e.g. population turnover per generation between the two ancestral populations. Some level of migration between the ancestral populations is required to permit coalescence to a single root per tree structure.

`seq_length` : int, default 1e4

The number of base pairs to simulate coalescence over, per window.

`recombination_rate` : float, default 1e-8

The recombination rate parameter for coalescence simulation in msprime.

`u` : float, default 1e-8

Defines the mutation rate. Used to convert diversity in terms of branch lengths into nucleotide diversity. 

`demographic_model` : string, default “one_pop”

Defines the demographic model to be used in extended coalescence simulation if rooted == True.

`rooted` : bool, default True

Only used when the task being performed involves coalescence simulation. If True, perform a second coalescent simulation using the tree sequence from coalescence simulation over the pedigree as a base, thereby rooting all sample individuals somewhere in the remote past before the explicit simulation.


