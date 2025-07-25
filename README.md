# Logistic Regression on Digits Dataset

A simple classification project using logistic regression on the scikit-learn digits dataset, extended with experiments on how label noise affects model performance.

## Overview

This project trains a baseline model to classify handwritten digits (0–9) using 8×8 grayscale images. It then introduces varying levels of random label noise (10% to 100%) to analyze how corrupted training labels impact classification accuracy and confusion matrices.

## Dataset

- **Source**: [`sklearn.datasets.load_digits()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- **Features**: Flattened 8×8 grayscale images (64 features per sample)
- **Labels**: Digits 0 through 9 (10 classes)

## Model

- **Algorithm**: Logistic Regression
  (`sklearn.linear_model.LogisticRegression`)
- **Noise Levels Tested**: 0%, 10%, 20%, 50%, 100%
- **Evaluation Metrics**:

  - Accuracy
  - Confusion Matrix
  - Accuracy vs. Label Noise Plot

## Results Snapshot

| Noise Level (%) | Accuracy |
| --------------- | -------- |
| 0               | \~0.95   |
| 10              | \~0.85   |
| 20              | \~0.88   |
| 50              | \~0.81   |
| 100             | \~0.01   |

(_Values may vary slightly depending on randomness._)

## Getting Started

1. Clone the repository
2. Set up a virtual environment
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:

   ```bash
   python label_noise_experiment.py
   ```

## Status

✅ Baseline model implemented
✅ Noise injection experiments completed
✅ Visualizations for accuracy and confusion matrices
✅ Automated noise-level experiments with loop refactor
