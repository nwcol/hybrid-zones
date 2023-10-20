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
	
`trial` will also include its own parameter set as an instance variable. If we wish to see part of the pedigree, rows can be printed with 

	print(pedigree_table.cols)
	
Single generations can be extracted out of the whole pedigree table using 

    generation_table = pedigree_table.get_generation(t)

	
discuss arrays

discuss genetic simulation


## Data structure




## Model structure






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