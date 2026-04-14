import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from dataset_loader import MRIDataset
from autoencoder import Encoder, Decoder, Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 🔥 TRAIN + TEST SPLIT
train_dataset = MRIDataset(r"D:\austism 4 trail\archive (1)\ABIDE\Combined Data", split="train")
test_dataset = MRIDataset(r"D:\austism 4 trail\archive (1)\ABIDE\Combined Data", split="test")

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

encoder = Encoder().to(device)
decoder = Decoder().to(device)
classifier = Classifier().to(device)

optimizer = optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()) +
    list(classifier.parameters()),
    lr=1e-4
)

print("\n--- Training ---")

for epoch in range(50):
    encoder.train()
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        z = encoder(x)
        recon = decoder(z)
        output = classifier(z)

        loss_cls = F.cross_entropy(output, y)
        loss_recon = F.mse_loss(recon, x)

        loss = loss_cls + 0.2 * loss_recon

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(output, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = 100 * correct / total

    # 🔥 VALIDATION
    encoder.eval()
    classifier.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            z = encoder(x)
            output = classifier(z)

            preds = torch.argmax(output, dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # 🔥 CHECKPOINT EVERY 5 EPOCHS
    if epoch % 10 == 0:
        torch.save(encoder.state_dict(), f"encoder_epoch_{epoch}.pth")
        torch.save(classifier.state_dict(), f"classifier_epoch_{epoch}.pth")
        print(f"Checkpoint saved at epoch {epoch}")


# 🔥 FINAL SAVE
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(classifier.state_dict(), "classifier.pth")

print("\nFinal models saved.")