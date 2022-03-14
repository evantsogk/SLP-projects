#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

EV_v3=${CURRENT_DIRECTORY}/../fsts/EV_v3.binfst
W_opensubtitles=${CURRENT_DIRECTORY}/../fsts/W_opensubtitles.binfst
EVW_v3=${CURRENT_DIRECTORY}/../fsts/EVW_v3.binfst

# input and output
WORDSYMS=${CURRENT_DIRECTORY}/../vocab/opensubtitles.syms

# create W
python3 mkfst_W_opensubtitles.py > ${CURRENT_DIRECTORY}/../fsts/W_opensubtitles.fst
# compile W
fstcompile --isymbols=${WORDSYMS} --osymbols=${WORDSYMS} ${CURRENT_DIRECTORY}/../fsts/W_opensubtitles.fst ${W_opensubtitles}

# compose EV spell checker with language model W
fstcompose ${EV_v3} ${W_opensubtitles} > ${EVW_v3}

