## Introduction

Hybzones is a population model for the study of the genetic structure of interspecies hybrid zones. It explicitly, although simply, models the behavior of individual organisms in a continuous 1-dimensional space across discrete generations to generate large, biologically plausible pedigrees. I have implemented various models of assortative mate selection and spatial fitness to further increase the biological realism of the system. 

Once pedigrees have been produced by the explicit simulation, they may be converted into msprime pedigrees using the PedigreeBuilder class and used as platforms for coalescent simulations. This pipeline allows us to examine a number of genetic statistics under varying explicit simulation parameters and models.


## Setup

put directions for setting up here


## Examples

Running an explicit simulation is relatively simple. Let us initialize and run a trial with the default parameters lasting 100 generations. The n_snaps argument will print a figure with 10 plots displaying genotype densities across space at 10 even intervals in time.

	params = parameters.Params(g=100)
	trial = Trial(params, n_snaps=10)
	
Upon initializion, the `Trial` class instance immediately runs the simulation. We might expect the simulation to take roughly 0.1-0.15 seconds per generation with a carrying capacity parameter of `K` = 10,000. The constructed pedigree table is an instance variable of the `trial` class instance; we can access it with

	pedigree_table = trial.pedigree_table
	
`trial` will also include its own parameters as an instance variable. If we wish to see part of the pedigree, rows can be printed with 

	print(pedigree_table.cols)
	
Single generations can be extracted out of the whole pedigree table using 

    generation_table = pedigree_table.get_generation(t)

	
discuss arrays

discuss genetic simulation


## Data structure

The data structure was inspired by the tskit Tables class and its subclasses. It was also designed to minimize memory utilization, as pedigree tables can become extremely large structures when simulations endure for long periods (tens of thousands of generations).

### Params class

The `Params` class holds the parameters which define the duration of the simulation, the models of mating and dispersal, and all other user-changeable simulation parameters. Further information is provided in the docs folder.

### Columns class

The `Columns` class is the basis of my table classes. It is composed of several long vectors of equal length, which act as columns in an array of heterogenous type. For instance, the `sex` column has the data type `np.uint8` while `x` is type `np.float32`. The number of columns included in a `Columns` instance is dynamic, but 4 column categories are mandatory since they are required to reconstruct ancestries: `id`, `maternal_id`, `paternal_id` and `time`. 

### Table class

The `Table` class is the superclass of the `PedigreeTable` and `GenerationTable` classes and acts as a wrapper for a single `Columns` class. Arbitrary slices of `GenerationTable` and `PedigreeTable` instances are returned as `Table` instances.

### GenerationTable class

The `GenerationTable` class is the structure which is actively manipulated throughout the process of simulation. It represents the present generation.

### PedigreeTable class

The product of a simulation. 

### Trial class


## Model: Process and Assumptions






## Final touches to implement:
- Finish documentation

- Implement migration properly (flux edge)

- add __repr__ for the base table class, pay some attention to this class

- better __repr__, __str__ for pedigree and generation tables

- check all the __repr__ and make sure they make sense

- fix all the functions/methods which save files to make sure they save in the correct directories

- handling zero length columns

- handling extinction and ungraceful exits for simulation

- examples?

- parameters doc

- set up tests