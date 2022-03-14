#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

# input and output
CHARSYMS=${CURRENT_DIRECTORY}/../vocab/chars.syms

# create fst
python3 mkfst_E.py > ${CURRENT_DIRECTORY}/../fsts/E.fst
# compile fst
fstcompile --isymbols=${CHARSYMS} --osymbols=${CHARSYMS} ${CURRENT_DIRECTORY}/../fsts/E.fst ${CURRENT_DIRECTORY}/../fsts/E.binfst

