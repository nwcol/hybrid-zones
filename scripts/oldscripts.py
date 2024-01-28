"""
For CHTC server trials. Interprets params files and uses them to run a
population simulation under defined parameters, then process it as specified

:param 1: cluster id
:param 2: process id
"""

import sys

import os

from diploid import pop_model

from diploid import genetic_model

from diploid.parameters import Params


test = True

if test:
    cluster_id = 0
    process_id = 0

else:
    cluster_id = int(sys.argv[1])
    process_id = int(sys.argv[2])


def get_param_filename(process_id):
    """
    Take the modulo of the process_id by the number of parameter files in
    the directory "param_filenames"; load the parameter file at this index.
    All parameter files in this directory are used. This allows cycling over
    a range of parameters. Take care to empty the directory of any
    undesired parameter files!

    :param process_id:
    :type process_id: int
    :return: relative path to the selected parameter file
    :return type: string
    """
    param_filenames = os.listdir("diploid/param_files")
    param_filenames.sort()
    i = process_id % len(param_filenames)
    param_filename = param_filenames[i]
    return param_filename


def get_pedigree_arr(params, filename_stem, id_string):
    """
    Run a trial and save the complete, bare pedigree array

    :param params: Params class instance
    :param filename_stem: the name of the loaded parameter .json file
    :param id_string: format "Cluster id"_"Process id"
    :return: None
    """
    trial = pop_model.Trial(params)
    filename = filename_stem + "_pedigree_arr_" + id_string + ".txt"
    trial.pedigree.save_txt(filename)


def get_subpop_arr(params, filename_stem, id_string):
    """
    Run a trial using the SubpopArr history type and save the bare subpop
    array as a .txt file.

    :param params: Params class instance
    :param filename_stem: the name of the loaded parameter .json file
    :param id_string: format "Cluster id"_"Process id"
    :return: None
    """
    if params.history_type != "SubpopArr":
        params.history_type = "SubpopArr"
        print("params.history_type changed to SubpopArr")
    trial = pop_model.Trial(params)
    filename = filename_stem + "_subpop_arr_" + id_string + ".txt"
    trial.subpop_arr.save_txt(filename)


def get_multi_window(params, filename_stem, id_string):
    """
    Generate an abbreviated pedigree and run a series of coalescence
    simulations over a single sub-pedigree sampled from it

    :param params: Params class instance
    :param filename_stem: the name of the loaded parameter .json file
    :param id_string: format "Cluster id"_"Process id"
    :return: None
    """
    if params.history_type != "AbbrevPedigree":
        params.history_type = "AbbrevPedigree"
        print("params.history_type changed to AbbrevPedigree")
    trial = pop_model.Trial(params)
    sample = genetic_model.AbbrevSamplePedigree.from_trial(trial)
    windows = genetic_model.MultiWindow.from_abbrev_pedigree(sample)
    windows.save_all(prefix=filename_stem, suffix=id_string)


script_dict = {"get_pedigree_arr": get_pedigree_arr,
               "get_subpop_arr": get_subpop_arr,
               "get_multi_window": get_multi_window}


def main(cluster_id, process_id):
    """
    Load a parameter instance and interpret its instructions; run the
    appropriate script.

    :return: int
    """
    param_filename = get_param_filename(process_id)
    long_param_filename = r"diploid/param_files/" + param_filename
    params = Params.load(long_param_filename)
    filename_stem = param_filename.replace(".json", "")
    id_string = str(cluster_id) + "_" + str(process_id)
    script_dict[params.task](params, filename_stem, id_string)
    print("params.task successfully executed")
    return 0


main(cluster_id, process_id)