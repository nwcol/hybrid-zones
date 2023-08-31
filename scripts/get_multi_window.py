from context import pedigrees

from context import genetics


def main(params, name_stem, number):
    trial = pedigrees.Trial(params)
    sample_pedigree = genetics.SamplePedigreeTable.from_trial(trial)
    multi_window = genetics.MultiWindow.new(sample_pedigree)
    multi_window.save(name_stem, number)
    return 0
