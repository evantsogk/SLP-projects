#!/usr/bin/env bash


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

EV=${CURRENT_DIRECTORY}/../fsts/EV.binfst
W=${CURRENT_DIRECTORY}/../fsts/W.binfst
EVW=${CURRENT_DIRECTORY}/../fsts/EVW.binfst

# compose EV spell checker with language model W
fstcompose ${EV} ${W} > ${EVW}

