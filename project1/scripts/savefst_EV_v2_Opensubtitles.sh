#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)
E_v2=${CURRENT_DIRECTORY}/../fsts/E_v2.binfst
V_opensubtitles=${CURRENT_DIRECTORY}/../fsts/V_opensubtitles.binfst
EV_v2_opensubtitles=${CURRENT_DIRECTORY}/../fsts/EV_v2_opensubtitles.binfst

# input and output
CHARSYMS=${CURRENT_DIRECTORY}/../vocab/chars.syms
WORDSYMS=${CURRENT_DIRECTORY}/../vocab/opensubtitles.syms

# create simpsons lexicon
# create fst
python3 mkfstOpensubtitlesLexicon.py > ${CURRENT_DIRECTORY}/../fsts/V_opensubtitles.fst
# compile fst
fstcompile --isymbols=${CHARSYMS} --osymbols=${WORDSYMS} ${CURRENT_DIRECTORY}/../fsts/V_opensubtitles.fst ${CURRENT_DIRECTORY}/../fsts/V_opensubtitles_init.binfst
# optimize fst
fstrmepsilon ${CURRENT_DIRECTORY}/../fsts/V_opensubtitles_init.binfst | fstdeterminize | fstminimize > ${V_opensubtitles}

# compose E fst with lexicon fst to create the EV fst
fstarcsort --sort_type=olabel ${E_v2} | fstcompose - ${V_opensubtitles} > ${EV_v2_opensubtitles}

