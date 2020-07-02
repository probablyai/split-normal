#!/bin/bash

# conda skeleton pypi split-normal

cd recipe
conda build --output-folder build -c conda-forge .

cd build
conda convert --platform all linux-64/*.tar.bz2 -o .