#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

V=${CURRENT_DIRECTORY}/../fsts/V.binfst
W=${CURRENT_DIRECTORY}/../fsts/W.binfst
VW=${CURRENT_DIRECTORY}/../fsts/VW.binfst

# input and output
WORDSYMS=${CURRENT_DIRECTORY}/../vocab/words.syms

# compose W with V
fstcompose ${V} ${W} > ${VW}


# draw fst
fstdraw --isymbols=${WORDSYMS} --osymbols=${WORDSYMS} ${CURRENT_DIRECTORY}/../fsts/VW.binfst ${CURRENT_DIRECTORY}/../fsts/VW.dot
dot -Tpng ${CURRENT_DIRECTORY}/../fsts/VW.dot > ${CURRENT_DIRECTORY}/../fsts/VW.png

