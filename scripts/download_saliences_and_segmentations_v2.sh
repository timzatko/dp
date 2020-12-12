#!/bin/bash

mkdir -p tmp
cd tmp || exit
gdown https://drive.google.com/uc?id=1tfQ5LdV1MvRWIdvptE1lyRBiUayrmonT --output saliences_and_segmentations_v2.zip
unzip saliences_and_segmentations_v2s.zip