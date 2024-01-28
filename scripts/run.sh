#!/bin/bash

tar -xzf python39.tar.gz
tar -xzf packages.tar.gz

export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD

python3 hybzones/scripts/interpret.py $1 $2