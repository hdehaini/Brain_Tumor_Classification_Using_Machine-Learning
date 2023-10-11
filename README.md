# Brain Tumor Classification using Machine Learning

This repository contains the code and resources for a machine learning project that classifies brain MRI images as either tumor or non-tumor based on a feature dataset including first-order and texture features.

## Overview

Brain tumor diagnosis is a critical medical task, and this project leverages machine learning to assist in the classification of MRI images. The primary goal is to provide a reliable tool for early tumor detection.

## Features

- Preprocessing of the dataset
- Training a neural network for classification
- Evaluation metrics and visualization
- Inference functionality for new data
- Hyperparameter tuning (optional)
- Model deployment (optional)

## Dataset

The dataset used in this project consists of MRI images with the following features:

- First Order Features: Mean, Variance, Standard Deviation, Skewness, Kurtosis
- Second Order Features: Contrast, Energy, ASM (Angular second moment), Entropy, Homogeneity, Dissimilarity, Correlation, Coarseness

The 'Class' column in the dataset specifies whether the image contains a tumor (1) or not (0).

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/hdehaini/Brain_Tumor_Classification_Using_Machine-Learning.git
   cd Brain_Tumor_Classification_Using_Machine-Learning
