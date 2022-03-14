#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

# input and output
WORDSYMS=${CURRENT_DIRECTORY}/../vocab/words.syms

# create fst
python3 mkfst_W.py > ${CURRENT_DIRECTORY}/../fsts/W.fst
# compile fst
fstcompile --isymbols=${WORDSYMS} --osymbols=${WORDSYMS} ${CURRENT_DIRECTORY}/../fsts/W.fst ${CURRENT_DIRECTORY}/../fsts/W.binfst
# draw fst
fstdraw --isymbols=${WORDSYMS} --osymbols=${WORDSYMS} ${CURRENT_DIRECTORY}/../fsts/W.binfst ${CURRENT_DIRECTORY}/../fsts/W.dot
dot -Tpng ${CURRENT_DIRECTORY}/../fsts/W.dot > ${CURRENT_DIRECTORY}/../fsts/W.png

