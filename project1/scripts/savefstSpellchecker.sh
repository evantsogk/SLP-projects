#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

LEVENSHTEIN=${CURRENT_DIRECTORY}/../fsts/L.binfst
LEXICON=${CURRENT_DIRECTORY}/../fsts/V.binfst
SPELLCHECKER=${CURRENT_DIRECTORY}/../fsts/S.binfst

# compose levenshtein fst with lexicon fst to create the spellchecker fst
fstarcsort --sort_type=olabel ${LEVENSHTEIN} | fstcompose - ${LEXICON} > ${SPELLCHECKER}

