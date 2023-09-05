from context import pedigrees


def main(params, name_stem, number):
    trial = pedigrees.Trial(params)
    filename = name_stem + "_pedigree_" + number + ".txt"
    trial.pedigree_table.cols.save_txt(filename)
    return 0