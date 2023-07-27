"""
Utilities for loading directories and large sets of data
"""
import numpy as np

import sys

import os

from diploid import pop_model

from diploid.parameters import Params


def load_subpop_arrs(directory):
    """Load several SubpopArr files from a directory and return them in a list
    """
    filenames = os.listdir(directory)
    full_filenames = [directory + r"/" + filename for filename in filenames]
    subpop_arrs = []
    for filename in full_filenames:
        subpop_arrs.append(pop_model.SubpopArr.load_txt(filename))
    return subpop_arrs


def parse_subpop_arrs(subpop_arrs):
    """Check to see if all subpop_arrs have the same bin_size, length in
    generations, and parameter sets to insure that they are commensurable
    """
    pass # write later


