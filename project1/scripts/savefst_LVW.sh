#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

LV=${CURRENT_DIRECTORY}/../fsts/S.binfst
W=${CURRENT_DIRECTORY}/../fsts/W.binfst
LVW=${CURRENT_DIRECTORY}/../fsts/LVW.binfst

# compose LV spell checker with language model W
fstcompose ${LV} ${W} > ${LVW}

