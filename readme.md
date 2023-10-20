## Introduction

Hybzones is a population model for the study of the genetic structure of interspecies hybrid zones. It explicitly, although simply, models the behavior of individual organisms in a continuous 1-dimensional space across discrete generations to generate large, biologically plausible pedigrees. I have implemented various models of assortative mate selection and spatial fitness to further increase the biological realism of the system. 

Once pedigrees have been produced by the explicit simulation, they may be converted into tskit pedigrees for further genetic simulation in msprime. This pipeline allows us to examine a number of genetic statistics to be examined under a variety of explicit simulation parameters and models.


## Setup

put directions for setting up here


## Examples

Running an explicit simulation is relatively simple. Let us initialize and run a trial with the default parameters lasting 100 generations. The n_snaps argument will print a figure with 10 plots displaying genotype densities across space at 10 even intervals in time.

	params = parameters.Params(g=100)
	trial = Trial(params, n_snaps=10)




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