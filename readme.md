## Introduction

Hybzones is a model I wrote to study genetic structure in interspecies hybrid zones. It explicitly, although simply, models the behavior of individual organisms in a continuous 1-dimensional space across discrete generations to generate large, biologically plausible pedigrees. I have implemented various models of assortative mate selection and spatial fitness to further increase the biological realism of the system. 

Once pedigrees have been produced by the explicit simulation, they can be converted into msprime pedigrees using that packages' PedigreeBuilder class and used as platforms for coalescent simulations. This pipeline allows the examination of a number of genetic statistics under varying simulation parameters and models.


## Setup



## Examples

Running an explicit simulation is simple. Let us initialize and run a 100-generation trial with the default parameters. The n_snaps argument will print a figure with 10 plots displaying genotype densities across space at 10 even intervals in time.

	from hybzones import parameters, pedigrees
	params = parameters.Params(g=100)
	trial = pedigrees.Trial(params, n_snaps=10)
	
Upon initializion, the `Trial` class instance immediately runs the simulation. We might expect the simulation to take roughly 0.1-0.15 seconds per generation with a carrying capacity parameter of `K` = 10,000. The constructed pedigree table is an instance variable of the `trial` class instance; we can access it with

	pedigree_table = trial.pedigree_table
	
`trial` will also include its own parameters as an instance variable. If we wish to see part of the pedigree, rows can be printed with 

	print(pedigree_table.cols)
	
Single generations can be extracted out of the whole pedigree table using 

    generation_table = pedigree_table.get_generation(t)
	
Pedigrees can be saved as .ped files using

	pedigree_table.save_ped("filename.ped")

This will also save a .dat file recording associated parameters. There are two primary data structures for tersely representing population histories: genotype and allele density arrays. These structures record the abundance of genotypes or alleles in each generation and in a set of spatial bins. They can be generated from pedigree tables using

	from hybzones import arrays
	genotype_arr = arrays.GenotypeArr.from_pedigree(pedigree_table)
	allele_arr = arrays.AlleleArr.from_pedigree(pedigree_table)

Pedigrees can be 

	import numpy as np
	from hybzone import genetics
	bin_edges = np.linspace(0, 1, 11)
	sample_size = 10
	sample_pedigree = genetics.SamplePedigreeTable.sample(pedigree_table, bin_edges, sample_size)
	
sss

	windows = genetics.MultiWindow(sample_pedigree, 10)
	windows.mean_pi
	windows.mean_pi_xy
	

	


## Data structure

The data structure was inspired by the tskit Tables class and its subclasses. It was also designed to minimize memory utilization, as pedigree tables can become extremely large when simulations endure for long periods (tens of thousands of generations).

### Params class

The `Params` class holds the parameters which define the duration of the simulation, the mating and dispersal models, and all other user-changeable simulation parameters. Further information is provided in the docs directory.

### Columns class

The `Columns` class provides the basic structure of pedigree-like objects, including pedigree tables and generation tables. It is composed of several long vectors of equal length, which act as columns in an array of heterogenous type. For instance, the `sex` column has the data type `np.uint8` while the position `x` has type `np.float32`. The number of columns included in a `Columns` instance is dynamic, but 4 column categories are mandatory since they are required to reconstruct ancestries: `id`, `maternal_id`, `paternal_id` and `time`. 

### Table class

The `Table` class is the superclass of the `PedigreeTable` and `GenerationTable` classes and acts as a wrapper for a single `Columns` class. Arbitrary slices of `GenerationTable` and `PedigreeTable` instances are returned as `Table` instances.

### GenerationTable class

The `GenerationTable` class is the structure which is actively manipulated throughout the process of simulation. It represents a single generation, usually of still-living organisms. 

### PedigreeTable class

The `PedigreeTable` class holds complete pedigrees, recording the characteristics of all the organisms that existed throughout the course of a simulation. 

### Trial class

A `Trial` class instance has a `PedigreeTable` instance as an attribute, and upon initiation immediately executes the simulation loop to fill the pedigree. 

### SamplePedigreeTable class


### MultiWindow

