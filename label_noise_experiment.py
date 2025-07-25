# Import libraries for data handling, model training, evaluation, and visualization
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better-looking plots
sns.set_theme()

from sklearn.datasets import load_digits

# Load the digits dataset (8×8 grayscale images of handwritten digits 0–9)
# Each sample is a flattened 8×8 = 64-feature vector
digits = load_digits()
X = digits.data
y = digits.target

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

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