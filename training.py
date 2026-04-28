import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from dataset_loader import MRIDataset
from autoencoder import Encoder, Decoder, Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# DATA
train_dataset = MRIDataset(r"D:\austism 4 trail\archive (1)\ABIDE\Combined Data", split="train")
test_dataset = MRIDataset(r"D:\austism 4 trail\archive (1)\ABIDE\Combined Data", split="test")

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# MODELS
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
    decoder.train()
    classifier.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    # -------- TRAIN --------
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

        train_loss += loss.item()

        preds = torch.argmax(output, dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    train_acc = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    # -------- VALIDATION --------
    encoder.eval()
    decoder.eval()
    classifier.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            z = encoder(x)
            recon = decoder(z)
            output = classifier(z)

            loss_cls = F.cross_entropy(output, y)
            loss_recon = F.mse_loss(recon, x)

            loss = loss_cls + 0.2 * loss_recon
            val_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(test_loader)

    print(f"Epoch {epoch} | "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # -------- SAVE EVERY 10 EPOCHS --------
    if (epoch + 1) % 10 == 0:
        torch.save(encoder.state_dict(), f"encoder_epoch_{epoch+1}.pth")
        torch.save(classifier.state_dict(), f"classifier_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")

# FINAL SAVE
torch.save(encoder.state_dict(), "encoder_final.pth")
torch.save(classifier.state_dict(), "classifier_final.pth")

print("\nFinal models saved.")