#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)
E_v2=${CURRENT_DIRECTORY}/../fsts/E_v2.binfst
V_simpsons=${CURRENT_DIRECTORY}/../fsts/V_simpsons.binfst
EV_v2_simpsons=${CURRENT_DIRECTORY}/../fsts/EV_v2_simpsons.binfst

# input and output
CHARSYMS=${CURRENT_DIRECTORY}/../vocab/chars.syms
WORDSYMS=${CURRENT_DIRECTORY}/../vocab/simpsons.syms

# create simpsons lexicon
# create fst
python3 mkfstSimpsonsLexicon.py > ${CURRENT_DIRECTORY}/../fsts/V_simpsons.fst
# compile fst
fstcompile --isymbols=${CHARSYMS} --osymbols=${WORDSYMS} ${CURRENT_DIRECTORY}/../fsts/V_simpsons.fst ${CURRENT_DIRECTORY}/../fsts/V_simpsons_init.binfst
# optimize fst
fstrmepsilon ${CURRENT_DIRECTORY}/../fsts/V_simpsons_init.binfst | fstdeterminize | fstminimize > ${V_simpsons}

# compose E fst with lexicon fst to create the EV fst
fstarcsort --sort_type=olabel ${E_v2} | fstcompose - ${V_simpsons} > ${EV_v2_simpsons}

