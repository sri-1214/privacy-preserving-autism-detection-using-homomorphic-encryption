import torch
import numpy as np

from dataset_loader import MRIDataset
from autoencoder import Encoder, Classifier

# ======================
# LOAD MODELS
# ======================
encoder = Encoder()
classifier = Classifier()

encoder.load_state_dict(torch.load("encoder.pth", map_location="cpu"))
classifier.load_state_dict(torch.load("classifier.pth", map_location="cpu"))

encoder.eval()
classifier.eval()

print("Models loaded successfully.\n")

# ======================
# LOAD FULL DATASET (ALL SAMPLES)
# ======================
dataset = MRIDataset(
    r"D:\austism 4 trail\archive (1)\ABIDE\Combined Data",
    split="train"   # train set
)

# also include test set
test_dataset = MRIDataset(
    r"D:\austism 4 trail\archive (1)\ABIDE\Combined Data",
    split="test"
)

# combine both
full_data = dataset.samples + test_dataset.samples

print(f"\nTotal samples for evaluation: {len(full_data)}")

# ======================
# EVALUATION
# ======================
correct = 0
wrong = 0

for i, (path, label) in enumerate(full_data):

    # reload through dataset logic
    x, y = dataset.__getitem__(i % len(dataset))  # reuse loader safely
    x = x.unsqueeze(0)

    with torch.no_grad():
        z = encoder(x)
        output = classifier(z)

    pred = torch.argmax(output, dim=1).item()

    if pred == y.item():
        correct += 1
    else:
        wrong += 1

# ======================
# RESULTS
# ======================
total = correct + wrong
accuracy = (correct / total) * 100

print("\n==========================")
print(f"Total Samples: {total}")
print(f"Correct Predictions: {correct}")
print(f"Wrong Predictions: {wrong}")
print(f"Accuracy: {accuracy:.2f}%")
print("==========================")