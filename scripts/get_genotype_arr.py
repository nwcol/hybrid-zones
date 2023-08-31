from context import pedigrees


def main(params, name_stem, number):
    if params.history_type != "genotype_arr":
        params.history_type = "genotype_arr"
        print("parameter history_type set to genotype_arr")
    trial = pedigrees.Trial(params)
    filename = name_stem + "_arr_" + number
    trial.genotype_arr.save_txt(filename)
    return 0
