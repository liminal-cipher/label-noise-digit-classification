# Import libraries for data handling, model training, evaluation, and visualization
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better-looking plots
sns.set_theme()
sns.set_palette("colorblind")

from sklearn.datasets import load_digits

# Load the digits dataset (8×8 grayscale images of handwritten digits 0–9)
# Each sample is a flattened 8×8 = 64-feature vector
digits = load_digits()
X = digits.data
y = digits.target

print(f"Number of samples: {X.shape[0]}")
print(f"Number of features per sample: {X.shape[1]}")
print(f"Number of label entries: {len(y)}")
print(f"Number of unique classes: {len(np.unique(y))}")

# Split into training and test sets (80/20), stratified to keep class balance
SEED = 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print("Number of training samples:", len(X_train))
print("Number of testing samples:", len(X_test))

# Train a baseline model with clean labels (no noise)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate baseline model
y_pred = model.predict(X_test)
acc_0 = accuracy_score(y_test, y_pred)
print(f"Accuracy with 0% noisy labels: {acc_0:.4f}")

# Plot baseline confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (0% Noisy Labels)")
plt.tight_layout()
plt.show()

# Function to add random label noise
def add_label_noise(y, noise_ratio, seed=0):
    """
    Randomly changes a given percentage of labels to incorrect values.
    """
    np.random.seed(seed)
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_ratio * n_samples)

    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    for idx in noisy_indices:
        original_label = y_noisy[idx]
        new_label = np.random.choice([l for l in range(10) if l != original_label])
        y_noisy[idx] = new_label

    return y_noisy

# === Train models with various levels of label noise ===

# 10% noise
y_train_10_noise = add_label_noise(y_train, 0.1)
model_10 = LogisticRegression(max_iter=3000)
model_10.fit(X_train, y_train_10_noise)

y_pred_10 = model_10.predict(X_test)
acc_10 = accuracy_score(y_test, y_pred_10)
print(f"Accuracy with 10% noisy labels: {acc_10:.4f}")

cm_10 = confusion_matrix(y_test, y_pred_10)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_10, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (10% Noisy Labels)")
plt.show()

# 20% noise
y_train_20_noise = add_label_noise(y_train, 0.2)
model_20 = LogisticRegression(max_iter=3000)
model_20.fit(X_train, y_train_20_noise)

y_pred_20 = model_20.predict(X_test)
acc_20 = accuracy_score(y_test, y_pred_20)
print(f"Accuracy with 20% noisy labels: {acc_20:.4f}")

cm_20 = confusion_matrix(y_test, y_pred_20)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_20, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (20% Noisy Labels)")
plt.show()

# 50% noise
y_train_50_noise = add_label_noise(y_train, 0.5)
model_50 = LogisticRegression(max_iter=3000)
model_50.fit(X_train, y_train_50_noise)

y_pred_50 = model_50.predict(X_test)
acc_50 = accuracy_score(y_test, y_pred_50)
print(f"Accuracy with 50% noisy labels: {acc_50:.4f}")

cm_50 = confusion_matrix(y_test, y_pred_50)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_50, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (50% Noisy Labels)")
plt.show()

# 100% noise
y_train_100_noise = add_label_noise(y_train, 1.0)
model_100 = LogisticRegression(max_iter=3000)
model_100.fit(X_train, y_train_100_noise)

y_pred_100 = model_100.predict(X_test)
acc_100 = accuracy_score(y_test, y_pred_100)
print(f"Accuracy with 100% noisy labels: {acc_100:.4f}")

cm_100 = confusion_matrix(y_test, y_pred_100)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_100, annot=True, fmt='d', cmap='Reds', cbar=False, vmin=0, vmax=40)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (100% Noisy Labels)")
plt.show()

# === Plot overall performance vs. noise level ===
noise_levels = [0, 10, 20, 50, 100]
accuracies = [acc_0, acc_10, acc_20, acc_50, acc_100]

plt.figure(figsize=(8, 5))
plt.plot(noise_levels, accuracies, marker='o', linestyle='-', linewidth=2)

plt.xticks(noise_levels)
plt.xlabel("Label Noise Level (%)")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs. Label Noise Level")
plt.grid(True)

# Annotate accuracy values on plot
for x, y in zip(noise_levels, accuracies):
    plt.text(x, y + 0.01, f"{y:.2f}", ha='center', fontsize=9)

plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()