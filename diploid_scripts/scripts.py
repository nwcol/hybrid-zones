"""
For CHTC server trials. Interprets params files and uses them to run a
population simulation under defined parameters, then process it as specificed

Arguments
------------
c_id : int
    Cluster id

p_id : int
    Proc id
"""

import sys

import os

import time

import numpy as np

from diploid import pop_model

from diploid import genetic_model

from diploid.parameters import Params


time0 = time.time()

test = False

if test != True:
    cluster_id = int(sys.argv[1])
    process_id = int(sys.argv[2])
    param_filenames = os.listdir("diploid_model/param_files")
    param_filenames.sort()
    i = process_id % len(param_filenames)
    param_filename = param_filenames[i]

elif test == True:
    cluster_id = 10
    process_id = 0
    param_filename = "group1.json"

extended_param_filename = r"diploid_model/param_files/" + param_filename
params = Params.load(extended_param_filename)
filename_stem = param_filename.replace(".json", "")

reports.make_report("Param instance " + param_filename + " loaded")


def build_filename(out_type):
    filename = (filename_stem + "_" + out_type + "_" + str(cluster_id) + "_"
                + str(process_id) + ".txt")
    return (filename)


def return_pedigree():
    pedigree_data = dip.run(params, get_matrix=False, time0=time0)
    dip.save_pedigree(pedigree_data, build_filename("pedigree"))


def return_pop_matrix():
    pop_matrix = dip.run(params, get_matrix=True, time0=time0)
    dip.save_pop_matrix(pop_matrix, params, build_filename("popmatrix"))


def diversity_summary():
    """Get a summary of diverisity and divergence in 10 spatial bins
    """
    pedigree = dip.run(params, get_matrix=False, time0=time0)
    ts = gen.simulate_coalescence(pedigree, n_vec=params.coalesc_n_vec,
                                  x_ranges=params.coalesc_x_range)
    gen.save_arr(gen.diversity_summary(ts), params, build_filename("pi"))
    gen.save_arr(gen.group_diversity_summary(ts), params,
                 build_filename("pi_X"))
    gen.save_arr(gen.divergence_summary(ts), params, build_filename("pi_XY"))


def multi_window_diversity():
    """Run a bunch of diversity summaries on a given number of base pair
    windows
    """
    pedigree = dip.run(params, get_matrix=False, time0=time0)
    pi, pi_XY = gen.multi_window(pedigree, params.n_windows, params)
    gen.save_arr(pi, params, build_filename("pi"))
    gen.save_arr(pi_XY, params, build_filename("pi_XY"))


def unrooted_multi_window():
    pedigree = dip.run(params, get_matrix=False, time0=time0)
    pi, pi_XY = gen.unrooted_multi_window(pedigree, params)
    gen.save_arr(pi, params, build_filename("pi"))
    gen.save_arr(pi_XY, params, build_filename("pi_XY"))

try:
    func = eval(params.task)
    func()
    print("params.task successfully executed")
except:
    print("invalid params.task value!!!")

