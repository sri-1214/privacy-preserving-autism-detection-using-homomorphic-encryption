import matplotlib.pyplot as plt
import numpy as np

models = [
    "HE Logistic Regression",
    "Encrypted ML Classifier",
    "HE Cancer Predictor",
    "Secure Logistic Model",
    "Encrypted Predictive Model",
    "Proposed HE-CKD Model"
]

accuracy = [68.2, 70.5, 69.8, 67.4, 65.0, 72.68]
precision = [58.1, 60.3, 59.2, 57.5, 55.0, 62.82]
recall = [80.5, 82.0, 83.4, 79.6, 78.0, 88.31]

x = np.arange(len(models))
width = 0.25

plt.figure()

plt.bar(x - width, accuracy, width)
plt.bar(x, precision, width)
plt.bar(x + width, recall, width)

plt.xticks(x, models, rotation=25)
plt.ylabel("Percentage (%)")
plt.title("Performance Comparison of Privacy-Preserving HE Models")

plt.legend(["Accuracy", "Precision", "Recall"])

plt.tight_layout()
plt.show()