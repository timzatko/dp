# DP

The thesis is located in [thesis/main.pdf](./thesis/main.pdf).

Structure:
- `notebooks` - contains notebooks which I ran on my machine in local environment
- `colab_notebooks` - contains notebooks which I ran o Google Colaboratory
- `conda_notebooks` - contains notebooks which I ran in Conda environment

## General Setup

### Prerequisites

- [Python](https://www.python.org/) (for data download)

### Setup

1. `pip install gdown`
1. Download data from `https://drive.google.com/file/d/1JJIqPQJfwR7GCvwtjtSrAHGY6II0AOTt` and extract them to `tmp/saliencies_and_segmentations_v2`. Run: `sh ./scripts/download_saliences_and_segmentations_v2.sh` 

## Conda Notebooks

### Prerequisites

- [Anaconda](https://anaconda.com/anaconda/install)

### Installation

1. `conda env create -f environment.yml`
