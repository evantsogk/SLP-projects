#!/usr/bin/env bash

# Creates EV_v3 spellchecker with add-k smoothing

# Constants.
CURRENT_DIRECTORY=$(dirname $0)
E_v3=${CURRENT_DIRECTORY}/../fsts/E_v3.binfst
V=${CURRENT_DIRECTORY}/../fsts/V_opensubtitles.binfst
EV_v3=${CURRENT_DIRECTORY}/../fsts/EV_v3.binfst

# input and output
CHARSYMS=${CURRENT_DIRECTORY}/../vocab/chars.syms

# create fst
python3 mkfst_E_v3.py > ${CURRENT_DIRECTORY}/../fsts/E_v3.fst
# compile fst
fstcompile --isymbols=${CHARSYMS} --osymbols=${CHARSYMS} ${CURRENT_DIRECTORY}/../fsts/E_v3.fst ${E_v3}
# compose E fst with lexicon fst to create the EV fst
fstarcsort --sort_type=olabel ${E_v3} | fstcompose - ${V} > ${EV_v3}

