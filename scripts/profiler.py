from memory_profiler import profile

import numpy as np

from context import parameters

from context import pedigrees

from context import genetics


@profile
def main():
    # trial = pedigrees.Trial.new(10_000, 10)
    trial = np.zeros((100, 100))
    return trial


main()
