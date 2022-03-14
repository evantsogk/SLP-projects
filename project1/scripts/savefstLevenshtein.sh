#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

# input and output
CHARSYMS=${CURRENT_DIRECTORY}/../vocab/chars.syms

# create fst
python3 mkfstLevenshtein.py > ${CURRENT_DIRECTORY}/../fsts/L.fst
# compile fst
fstcompile --isymbols=${CHARSYMS} --osymbols=${CHARSYMS} ${CURRENT_DIRECTORY}/../fsts/L.fst ${CURRENT_DIRECTORY}/../fsts/L.binfst
# draw fst
fstdraw --isymbols=${CHARSYMS} --osymbols=${CHARSYMS} ${CURRENT_DIRECTORY}/../fsts/L.binfst ${CURRENT_DIRECTORY}/../fsts/L.dot
dot -Tpng ${CURRENT_DIRECTORY}/../fsts/L.dot > ${CURRENT_DIRECTORY}/../fsts/L.png

