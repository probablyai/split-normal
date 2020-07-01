#!/bin/bash

# PyPI build
python build.py
python setup.py sdist

# PyPI test upload
python -m twine upload --repository testpypi dist/*

# PyPI test installation
pip install -i https://test.pypi.org/simple/ split-normal==0.1.0a1 --extra-index-url https://pypi.org/simple

# PyPI upload
python -m twine upload dist/*

# PyPI installation
pip install split-normal

# Conda
conda skeleton pypi split-normal
conda build -c conda-forge split-normal