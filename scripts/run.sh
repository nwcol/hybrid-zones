#!/bin/bash

tar -xzf python39.tar.gz
tar -xzf packages.tar.gz

export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD

python3 hybzones/scripts/get_multi_window.py $1 $2 