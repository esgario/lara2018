# Deep Learning for Classification and Severity Estimation of Coffee Leaf Biotic Stress

This repository is intended to develop the work supported by the Latin America Research Awards 2018.

The work is divided into two parts:

* [Classification](classification/README.md): This folder have the solution developed in the paper. Where we developed a multi-task model to classify the symptoms and estimate the severity of the biotic stress.
* [Segmentation](segmentation/README.md): This is an extension of the initial work, where we try to segment the symptoms in the images.

## Requirements

[Install anaconda.](https://docs.anaconda.com/anaconda/install/)

Create conda environment with python 3:
```
conda create -n <your_env_name> python=3.7
conda activate <your_env_name>
```

Then install the requirements:
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install Pillow==6.1 pandas matplotlib bokeh opencv
pip install -r requirements.txt
```

## Dataset

All images collected to build the datasets used in this work are public available at the following link:

[Coffee datasets](https://drive.google.com/open?id=15YHebAGrx1Vhv8-naave-R5o3Uo70jsm).

* Our experiments using the symptom datasets combine our images with those available at https://www.digipathos-rep.cnptia.embrapa.br/. However, the public dataset above contain only the images we collect, but here on github you will find exactly what was used in the experiments.

## Research

The development of this work led to the publication of an [paper](https://www.sciencedirect.com/science/article/pii/S0168169919313225) in Computers and Electronics in Agriculture journal.