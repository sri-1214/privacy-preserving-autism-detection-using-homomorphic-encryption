import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from dataset_loader import MRIDataset
from autoencoder import Encoder, Classifier

# LOAD MODELS
encoder = Encoder()
classifier = Classifier()

encoder.load_state_dict(torch.load("encoder_final.pth", map_location="cpu"))
classifier.load_state_dict(torch.load("classifier_final.pth", map_location="cpu"))
encoder.eval()
classifier.eval()

print("Models loaded.\n")

# LOAD TEST DATA (unseen)
dataset = MRIDataset(
    r"D:\austism 4 trail\archive (1)\ABIDE\Combined Data",
    split="test"
)

y_true = []
y_pred = []

print("Evaluating on full test dataset...\n")

for i in range(len(dataset)):
    x, y = dataset[i]
    x = x.unsqueeze(0)

    with torch.no_grad():
        z = encoder(x)
        output = classifier(z)

    pred = torch.argmax(output, dim=1).item()

    y_true.append(y.item())
    y_pred.append(pred)

# ======================
# METRICS
# ======================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("=========== RESULTS ===========")
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}")
print(f"Recall    : {recall*100:.2f}")
print("\nConfusion Matrix:")
print(cm)
print("==============================")