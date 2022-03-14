#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

# input and output
CHARSYMS=${CURRENT_DIRECTORY}/../vocab/chars.syms
WORDSYMS=${CURRENT_DIRECTORY}/../vocab/words.syms

# create fst
python3 mkfstLexicon.py > ${CURRENT_DIRECTORY}/../fsts/V.fst
# compile fst
fstcompile --isymbols=${CHARSYMS} --osymbols=${WORDSYMS} ${CURRENT_DIRECTORY}/../fsts/V.fst ${CURRENT_DIRECTORY}/../fsts/V_init.binfst
# optimize fst
fstrmepsilon ${CURRENT_DIRECTORY}/../fsts/V_init.binfst | fstdeterminize | fstminimize > ${CURRENT_DIRECTORY}/../fsts/V.binfst

