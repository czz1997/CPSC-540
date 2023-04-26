# Feature Extraction via Sparse Representation-based Classification Algorithms

UBC CPSC 540 Course Project

Group members: Zhongze Chen, Yibo Jiao

## Introduction

This repository implements a data-driven pipeline that combines feature extraction and sparse-representation-based classification.

## Usage

* Experiment on Yale Face dataset
  * Use `yalefaces-feat-extract.ipynb` to extract features
  * Use `yalefaces-recognition.ipynb` for classification
* Experiment on [Visual Product Recognition Challenge](https://www.aicrowd.com/challenges/visual-product-recognition-challenge-2023) dataset
  * Configure feature extractor using one of the `.yml` file in the `./config` folder
  * Use `VPRC2023-feat-extract.ipynb` to extract features with given feature extractor config
  * Use `VPRC2023-recognition.ipynb` for classification

## Acknowledgement

* Our code is built upon [Visual Product Recognition 2023 Starter Kit](https://gitlab.aicrowd.com/aicrowd/challenges/visual-product-matching-2023/visual-product-recognition-2023-starter-kit)