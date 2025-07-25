# Logistic Regression on Digits Dataset

A simple classification project using logistic regression on the scikit-learn digits dataset.

## Overview

This project trains a baseline model on clean (noise-free) data to classify handwritten digits (0–9) based on 8×8 grayscale images. The goal is to establish a performance benchmark before introducing label noise in future steps.

## Dataset

- Source: [`sklearn.datasets.load_digits()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- Features: Flattened 8×8 images (64 features)
- Classes: Digits 0 through 9

## Model

- Algorithm: Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- Evaluation: Accuracy and confusion matrix

## Status

✅ Initial baseline model complete  
⬜ Experiments with label noise (coming next)

## Getting Started

1. Clone the repository
2. Set up a virtual environment
3. Install dependencies from `requirements.txt`
4. Run `label_noise_experiment.py`
