import sys

from context import pedigrees

from context import genetics

import eval_params


cluster_id = int(sys.argv[1])
process_id = int(sys.argv[2])


def main():
    params, name_stem, number = eval_params.main(cluster_id, process_id)
    trial = pedigrees.Trial(params)
    sample_pedigree = genetics.SamplePedigreeTable.from_trial(trial)
    multi_window = genetics.MultiWindow.new(sample_pedigree)
    multi_window.save(name_stem, number)
    return 0


main()
