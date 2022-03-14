#!/usr/bin/env bash

# Creates EV_v2 spellchecker with add-1 smoothing

# Constants.
CURRENT_DIRECTORY=$(dirname $0)
E_v2=${CURRENT_DIRECTORY}/../fsts/E_v2.binfst
V=${CURRENT_DIRECTORY}/../fsts/V.binfst
EV_v2=${CURRENT_DIRECTORY}/../fsts/EV_v2.binfst

# input and output
CHARSYMS=${CURRENT_DIRECTORY}/../vocab/chars.syms

# create fst
python3 mkfst_E_v2.py > ${CURRENT_DIRECTORY}/../fsts/E_v2.fst
# compile fst
fstcompile --isymbols=${CHARSYMS} --osymbols=${CHARSYMS} ${CURRENT_DIRECTORY}/../fsts/E_v2.fst ${E_v2}
# compose E fst with lexicon fst to create the EV fst
fstarcsort --sort_type=olabel ${E_v2} | fstcompose - ${V} > ${EV_v2}

