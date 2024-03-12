### Note
I worked on this project during the 2022-2023 academic year as an undergraduate researcher. It is no longer under active development, but please feel free to contact me or open issues if you find that anything is obscure or incorrect in function.

### Introduction
"Hybzones" is a model I wrote in Python to study the structure of genetic variation in biological hybrid zones. It explicitly simulates the behavior of organisms under assortative mating regimes in a continuous 1-dimensional space across discrete generations to generate large, biologically plausible pedigrees, which can be transformed into `msprime` pedigree table collections for use in coalescent simulation. This allows the estimation of a variety of genetic parameters through space.

The model is focused on ecological interactions across a frontier or cline between interbreeding taxa, and follows in many respects the model of Payne and Krakauer (Sexual Selection, Space, and Speciation, International Journal of Organic Evolution 51:1 1997). Organisms, which are sexed and diploid, possess two fully penetrant, unlinked biallelic loci 'A' and 'B' which determine, respectively, a mating signal and mating preference phenotype. Under the various models of assortative mating, females with genotype B_x/B_x generally sample mates with signal genotype A_x/A_x with higher probability than they do A_x/A_y or A_y/A_y. The default initial state distributes homozygous organisms with matching signal/preference allele on each side of the space, such that the population usually develops as a pair of flanking homogeneous subpopulations centered around a dynamic hybrid zone.

There are several models of spatially variable fitness and assortative mate selection, which moduleate the strength of assortation and the behavior of heterozygotes, to allow the simulation of various biological scenarios. More information about models and model parameters can be found in docs/parameters.md.

### Setup
I suggest setting up a Python virtual environment. I use pip as a package manager.

    git clone https://github.com/nwcol/hybzones.git
    python -m venv .venv
    source .venv/bin/activate

You can install the required packages using

    pip install -r hybzones/requirements.txt

Then install `hybzones` from the cloned directory. 

    pip install hybzones/
 

### Examples
Running a simulation is simple. Let's initialize and run a 100-generation trial with the default parameters. The `n_snaps` argument will print a figure with 10 plots displaying genotype densities across space at 10 even intervals in time.

	from hybzones import parameters, pedigrees
	params = parameters.Params(g=100)
	trial = pedigrees.Trial(params, n_snaps=10)
	
Upon initializion, the `Trial` class instance immediately runs the simulation. The simulated pedigree is retained as a variable of the `trial` instance; we can access it with

	pedigree_table = trial.pedigree_table
	
If we wish to see part of the pedigree, rows can be printed with 

	print(pedigree_table.cols)
	
Single generations can be extracted out of the pedigree table using 

    generation_table = pedigree_table.get_generation(t)
	
Pedigrees can be saved as .ped files using

	pedigree_table.save_ped("filename.ped")

This will also save a .dat file recording associated parameters. There are two additional data structures for tersely representing population histories: genotype and allele density arrays. These structures record the abundance of genotypes or alleles in a set of spatial bins for each generation. They can be generated from pedigree tables using

	from hybzones import arrays
	genotype_arr = arrays.GenotypeArr.from_pedigree(pedigree_table)
	allele_arr = arrays.AlleleArr.from_pedigree(pedigree_table)

We can sample lineages from within pedigrees by instantiating a `SamplePedigreeTable`. This pedigree will record only a subset of the most recent generation, sampled through a given sampling configuration, and their ancestors. The `bin_edges` array specifies the spatial bins that we sample from, and `sample_size` specifies the number of organisms to sample per bin. A list or array of integers with length `len(bin_edges) - 1` may also be provided.

	import numpy as np
	from hybzones import genetics
	bin_edges = np.linspace(0, 1, 11)
	sample_size = 10
	sample_pedigree = genetics.SamplePedigreeTable.sample(pedigree_table, bin_edges, sample_size)
	
We can now run coalescent simulations through the sample pedigree. The `MultiWindow` class orchestrates the creation of an `msprime` pedigree table collection and coalescent simulation over a set of independent windows. The following command will run coalescent simulations in 10 10kb windows with a recombination rate `r=1e-8`.

	windows = genetics.MultiWindow(sample_pedigree, 10, seq_length=1e4, r=1e-8)
	
We can look at the average branch-length diversities and divergences across bins and windows using

    windows.mean_pi
	windows.mean_pi_xy

Or acess the whole array of mean diversities across windows with

    windows.pi

### Data structure
The data structure was inspired by the tskit Tables class and its subclasses. It was also designed to minimize memory utilization, as pedigree tables can become very large when simulations endure for thousands of generations.

#### Params class
The `Params` class holds the parameters which define the duration of the simulation, the mating and dispersal models, and all other user-changeable simulation parameters. Further information is provided in the docs directory.

#### Columns class
The `Columns` class provides the basic structure of pedigree-like objects, including pedigree tables and generation tables. It is composed of several long vectors of equal length, which act as columns in a table. The `sex` column is `np.uint8`,  position `x` has type `np.float32`, and organism ids are `np.int64`. The number of columns included in a `Columns` instance is dynamic, but 4 column categories are mandatory since they are required to reconstruct ancestries: `id`, `maternal_id`, `paternal_id` and `time`. 

#### Table class
The `Table` class is the superclass of the `PedigreeTable` and `GenerationTable` classes and acts as a wrapper for a single `Columns` class. Arbitrary slices of `GenerationTable` and `PedigreeTable` instances are returned as `Table` instances.

#### GenerationTable class
The `GenerationTable` class is the structure which is actively manipulated throughout the process of simulation. It represents a single generation, usually of still-living organisms. 

#### PedigreeTable class
The `PedigreeTable` class holds complete pedigrees, recording the characteristics of all the organisms that existed throughout the course of a simulation. 

#### Trial class
A `Trial` class instance has a `PedigreeTable` instance as an attribute, and upon initiation immediately executes the simulation loop to fill the pedigree. 
