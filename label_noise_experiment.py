# Import libraries for data handling, model training, evaluation, and visualization
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Style and palette
sns.set_theme()
sns.set_palette("colorblind")

from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

print(f"Number of samples: {X.shape[0]}")
print(f"Number of features per sample: {X.shape[1]}")
print(f"Number of label entries: {len(y)}")
print(f"Number of unique classes: {len(np.unique(y))}")

# Split into training and test sets
SEED = 0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")

# Function to add random label noise
def add_label_noise(y, noise_ratio, seed=0):
    """
    Randomly corrupt a given percentage of labels.
    """
    np.random.seed(seed)
    y_noisy = y.copy()
    n_noisy = int(noise_ratio * len(y_noisy))
    indices = np.random.choice(len(y_noisy), n_noisy, replace=False)
    for i in indices:
        orig = y_noisy[i]
        y_noisy[i] = np.random.choice([l for l in range(10) if l != orig])
    return y_noisy

# Mapping noise level → heatmap colormap
cmap_map = {
    0: "Blues",
    10: "Greens",
    20: "Purples",
    50: "Oranges",
    100: "Reds"
}

# Run experiments
noise_levels = [0, 10, 20, 50, 100]
accuracies = []

for lvl in noise_levels:
    ratio = lvl / 100
    # Prepare labels (0% just uses y_train unchanged)
    y_train_noisy = y_train if lvl == 0 else add_label_noise(y_train, ratio)
    # Train
    model = LogisticRegression(max_iter=1000 if lvl == 0 else 3000)
    model.fit(X_train, y_train_noisy)
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Accuracy with {lvl}% noisy labels: {acc:.4f}")
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                cmap=cmap_map[lvl], cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix ({lvl}% Noisy Labels)")
    plt.tight_layout()
    plt.show()

# Plot overall performance vs. noise level
plt.figure(figsize=(8, 5))
plt.plot(noise_levels, accuracies, marker='o', linestyle='-',
         linewidth=2)
plt.xticks(noise_levels)
plt.xlabel("Label Noise Level (%)")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs. Label Noise Level")
plt.grid(True)

for x, y in zip(noise_levels, accuracies):
    plt.text(x, y + 0.01, f"{y:.2f}", ha='center', fontsize=9)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()