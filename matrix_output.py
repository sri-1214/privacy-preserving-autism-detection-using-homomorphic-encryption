import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix values
cm = np.array([[321, 92],
               [46, 262]])

# Labels
labels = ["Typical Control (0)", "Autistic (1)"]

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix for ASD Classification")

plt.tight_layout()

# Save image (for report)
plt.savefig("confusion_matrix.png", dpi=300)

# Show plot
plt.show()