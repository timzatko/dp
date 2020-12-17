#!/bin/bash

mkdir -p tmp
cd tmp || exit
gdown https://drive.google.com/uc?id=1wrrxVBYTZgg8PC8-ITREP7EZHLeSHC9v --output saliences_and_segmentations_v2.zip
unzip saliences_and_segmentations_v2.zip