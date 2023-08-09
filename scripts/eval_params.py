import os

from context import parameters


def main(cluster_id, process_id, param_dir="parameters"):
    base_dir = os.getcwd()
    param_dir = base_dir + "/hybzones/parameters/"
    param_filenames = os.listdir(param_dir)
    param_filenames.sort()
    i = process_id % len(param_filenames)
    param_filename = param_filenames[i]
    params = parameters.Params.load(param_filename, param_dir)
    name_stem = param_filename.replace(".json", "")
    number = str(cluster_id) + "_" + str(process_id)
    return params, name_stem, number
