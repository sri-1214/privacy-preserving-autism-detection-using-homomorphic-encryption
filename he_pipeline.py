import torch
import numpy as np

from dataset_loader import MRIDataset
from autoencoder import Encoder, Classifier
from he_utils import create_context, encrypt_vector, decrypt_vector, encrypted_forward


# ======================
# LOAD MODELS
# ======================
encoder = Encoder()
classifier = Classifier()

encoder.load_state_dict(torch.load("encoder_final.pth", map_location="cpu"))
classifier.load_state_dict(torch.load("classifier_final.pth", map_location="cpu"))

encoder.eval()
classifier.eval()

print("Models loaded successfully.\n")


# ======================
# EXTRACT WEIGHTS
# ======================
def get_weights(model):
    w = model.fc.weight.data.numpy()
    b = model.fc.bias.data.numpy()
    return w, b


w, b = get_weights(classifier)


# ======================
# HE CONTEXT
# ======================
context = create_context()


# ======================
# LOAD FULL DATASET
# ======================
train_data = MRIDataset(
    r"D:\austism 4 trail\archive (1)\ABIDE\Combined Data",
    split="train"
)

test_data = MRIDataset(
    r"D:\austism 4 trail\archive (1)\ABIDE\Combined Data",
    split="test"
)

# Combine both
full_samples = train_data.samples + test_data.samples

print(f"\nTotal samples: {len(full_samples)}")


# ======================
# EVALUATION
# ======================
correct_plain = 0
correct_encrypted = 0
same_predictions = 0

total = len(full_samples)

print("\nRunning full evaluation...\n")

for i in range(total):

    # load sample properly
    if i < len(train_data):
        x, y = train_data[i]
    else:
        x, y = test_data[i - len(train_data)]

    x = x.unsqueeze(0)

    # -------- PLAINTEXT --------
    with torch.no_grad():
        z = encoder(x)
        plain_output = classifier(z)

    plain_pred = torch.argmax(plain_output, dim=1).item()

    # -------- ENCRYPTED --------
    z_np = z.squeeze().numpy()
    enc_z = encrypt_vector(context, z_np)

    enc_output = encrypted_forward(enc_z, w, b)
    decrypted_output = decrypt_vector(enc_output)

    enc_pred = int(np.argmax(decrypted_output))

    # -------- METRICS --------
    if plain_pred == y.item():
        correct_plain += 1

    if enc_pred == y.item():
        correct_encrypted += 1

    if plain_pred == enc_pred:
        same_predictions += 1

    # Optional: print progress every 100 samples
    if i % 100 == 0:
        print(f"Processed {i}/{total}")


# ======================
# FINAL RESULTS
# ======================
print("\n==========================")
print(f"Total Samples: {total}")
print(f"Plain Correct: {correct_plain}")
print(f"Encrypted Correct: {correct_encrypted}")
print(f"Wrong: {total - correct_plain}")

print(f"\nPlain Accuracy: {(correct_plain / total) * 100:.2f}%")
print(f"Encrypted Accuracy: {(correct_encrypted / total) * 100:.2f}%")
print(f"Prediction Match Rate: {(same_predictions / total) * 100:.2f}%")
print("==========================")