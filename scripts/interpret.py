import sys

import eval_params


cluster_id = int(sys.argv[1])
process_id = int(sys.argv[2])


def get_multi_window(params, name_stem, number):
    import get_multi_window
    get_multi_window.main(params, name_stem, number)


def get_genotype_arr(params, name_stem, number):
    import get_genotype_arr
    get_genotype_arr.main(params, name_stem, number)


def get_pedigree_table(params, name_stem, number):
    import get_pedigree_table
    get_pedigree_table.main(params, name_stem, number)


task_dict = {"get_multi_window": get_multi_window,
             "get_genotype_arr": get_genotype_arr,
             "get_pedigree_table": get_pedigree_table}


def main():
    params, name_stem, number = eval_params.main(cluster_id, process_id)
    task_dict[params.task](params, name_stem, number)


main()
