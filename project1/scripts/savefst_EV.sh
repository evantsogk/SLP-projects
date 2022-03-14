#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

E=${CURRENT_DIRECTORY}/../fsts/E.binfst
LEXICON=${CURRENT_DIRECTORY}/../fsts/V.binfst
EV=${CURRENT_DIRECTORY}/../fsts/EV.binfst

# compose E fst with lexicon fst to create the EV fst
fstarcsort --sort_type=olabel ${E} | fstcompose - ${LEXICON} > ${EV}

