## Introduction
The `Params` class a required argument for instantiating the `Trial` class. It stores an immutable collection of simulation parameters. Throughout simulation, the `Params` instance is repeatedly passed to lower-level objects and made an attribute of structures such as the pedigree table, where its attributes are accessed to govern a variety of simulation processes. Params instances can be saved as .JSON format files.

The only required argument to initialize a Params instance is `g`, the number of generations to simulate; all other attributes have default values which can be overridden by keyword arguments.

## Parameters
Here is a list of Params attributes describing their data types, biological interpretations, and roles in the simulation.

`subpop_n` : list of int, default `[N/2, 0, 0, 0, 0, 0, 0, 0, N/2]`

Defines the initial population sizes of each subpopulation, i.e. genotype. By default, 5000 individuals of each genotype A¹A¹B¹B¹, A²A²B²B² will compose the founding population.

`subpop_lims` : list of lists of int, default `[[0.0, 0.5], [], [], [], [], [], [], [], [0.5, 1.0]]`

Defines the limits in which each initial subpopulation exists. Individual positions are drawn uniformly at random from the defined intervals.

`K` : int, default `sum(subpop_n)`

Defines the carrying capacity of the entire space. Carrying capacity is assumed to be uniform, so that a subset of the space (x₀, x₁) has carrying capacity of K(x₁ - x₀).

`r` : float, default 0.5

Defines the population growth rate as a parameter in computing expected family size. Expected family sizes are density-dependent. Realized family sizes are drawn from a Poisson distribution.

`g` : int

Defines the duration of the simulation in generations.

`c` : float

Defines the strength of pure (homozygous preference to opposite homozygous signal) assortative mating. Specifically, `c` sets the ratio between the probability that a homozygous-preference female mates with an iso-signal homozygous-signal male and the probability that she mates with an allo-signal homozygous-signal male. The other assortative strengths are determined as a function of c by the preference model.

`pref_model` : string

Defines the preference model, which is a function from `c` to a 3x3 matrix which gives weights to the probabilities mate pairings by genotype.

`beta` : float, default 0.005

Defines the standard deviation of the distribution from which the mating probability masses are sampled. 

`bound` : float, default 0.02

Defines an upper bound on interaction- specifically mating- and on males’ perception of the signal phenotypes of other males.

`density_bound` : float, default 0.005

Defines the extent of space within which individuals affect each other’s observations of density. These observations control reproductive output, which is density-dependent.

`delta` : float

Defines the standard deviation of the normal distribution from which individual dispersal is sampled for all individuals in the random dispersal model, and for subsets of individuals in other dispersal models.

`dispersal_model` : string, default `“random”`

Defines the dispersal model. There are three dispersal models: `random`, `shift` and `scale`. The `random` dispersal model gives every individual a dispersal sampled randomly from a normal distribution with standard deviation delta.

The `shift` model samples from the same distribution for females and males with heterozygous signal genotype, but shifts the mean of the distribution for homozygous-signal males according to those males’ “perceptions” of the signal compositions to their left and right; the mean is shifted towards the direction with a higher proportion of iso-signal males, scaled proportional to the difference.

The `scale` model similarly treats homozygous-signal males differently, altering the standard deviations of their dispersal distributions. Their standard deviations are a function of the proportion of iso-signal males they observe about themselves; males with lower proportions will have higher standard deviations and will tend to disperse farther away.
Both nonrandom dispersal models propose a mechanism for males with lower mating chances to disperse into areas with potentially better mating prospects.

`edge_model` : string, default `“closed”`

Sort of a relic, from when there were multiple models for managing organisms that stray beyond the spatial boundaries `[0, 1]`. `"closed"` is the only fully implemented edge model, and it acts by setting the displacement of organisms whose movement would take them beyond the space to `0`, freezing them in place.

`scale_factor` : float, default 2

Defines the effect strength of the “scale” dispersal model. The scale dispersal model makes the standard deviation of each homozygous-signal males’ dispersal a function of `s`, where

	s = frequency of iso-signal males in neighborhood

The scale function is linear and passes through the points

	(s = 0, f(s) = scale_factor * delta), ( s= 1, f(s) = 1)

So that males with no iso-signal males in their neighborhood sample their dispersal with a standard deviation `scale_factor` times higher than those with entirely iso-signal males in their neighborhood. The scale function is defined as

	f(s) = ((1 - scale_factor) * s + scale_factor) * delta

`shift_factor` : float, default 2

Defines the effect strength for the “shift” dispersal model, which makes the mean of dispersal distance a function of the difference between iso-signal frequency in the neighborhoods to the left and right of homozygous-signal males. Concretely,

	s_r = frequency of iso-signal males to right 
	s_l = frequency of iso-signal males to left

	f(s) = (s_r - s_l) * shift_factor * delta

`intrinsic_fitness` : bool, default False

If True, exert an intrinsic fitness effect on heterozygote-signal individuals. 

`hyb_fitness` = float, default 1.0

If intrinsic_fitness is set to True, hyb_fitness gives the relative “innate” fitness of signal heterozygotes. Intrinsic fitness is so named because if `True`, it operates independently of spatial position as a uniform reduction in hybrid success. Whether intrinsic fitness affects females is governed by the same’

`extrinsic_fitness` : bool, default `False`

Defines whether environmental e.g. spatially variable fitness should operate on the population. Extrinsic fitness acts additively on the signal alleles. Fitness is simulated as a probability of survival, with a relative fitness of 1.0 guaranteeing survival. Reductions in fitness per allele are computed as logistic functions of spatial position `x`, parameterized using the `k_1`, `k_2`, `mid_1`, and `mid_2` parameters. Here are the fitness functions:

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

Defines the maximum reduction in fitness per allele. For a homozygous-signal individual, fitness at the opposite edge of space equals approximately `1 - mu * 2`.

`mid_1` : float, default 0.5

Defines the midpoint/inflection point of the fitness curve for A¹.

`mid_2` : float, default 0.5

Defines the midpoint/inflection point of the fitness curve for A². 
