import os
import json
import time

import torch
import torchaudio
from torch import nn, optim
from torch.utils.data import DataLoader

from model import SpeechCommandModel

CLASSES = ["up", "down", "yes", "no"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

DATA_ROOT = os.getenv("DATA_ROOT", "data")


class SubsetSpeechCommands(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root, subset):
        super().__init__(root=root, download=True, subset=subset)

        # Правильна побудова walker: беремо назву папки звуку через повний шлях
        new_walker = []
        for fileid in self._walker:
            path = os.path.join(self._path, fileid)
            label = path.split("/")[-2]  # ім’я папки (label)
            if label in CLASSES:
                new_walker.append(fileid)

        self._walker = new_walker

    def __getitem__(self, index):
        waveform, sample_rate, label, *_ = super().__getitem__(index)
        return waveform, CLASS_TO_IDX[label]


def get_dataloaders(batch_size=32):
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
    desired_len = 64

    def collate_fn(batch):
        specs = []
        labels = []
        for waveform, label in batch:
            # average по каналах
            waveform = waveform.mean(dim=0)
            spec = transform(waveform)  # [n_mels, time]

            if spec.shape[1] < desired_len:
                pad_amount = desired_len - spec.shape[1]
                spec = torch.nn.functional.pad(spec, (0, pad_amount))
            elif spec.shape[1] > desired_len:
                spec = spec[:, :desired_len]

            spec = spec.unsqueeze(0)  # [1, n_mels, time]
            specs.append(spec)
            labels.append(label)

        specs = torch.stack(specs)  # [B, 1, 64, 64]
        labels = torch.tensor(labels)
        return specs, labels

    train_ds = SubsetSpeechCommands(DATA_ROOT, subset="training")
    test_ds = SubsetSpeechCommands(DATA_ROOT, subset="testing")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dl, test_dl


def train(output_dir="artifacts", epochs=2):
    os.makedirs(os.path.join(output_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    log_path = os.path.join(output_dir, "logs", "train.log")
    metrics_path = os.path.join(output_dir, "metrics", "train_metrics.json")
    model_path = os.path.join(output_dir, "model", "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, test_dl = get_dataloaders(batch_size=32)

    model = SpeechCommandModel(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    history = []

    start_time = time.time()

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Start training\n")

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for X, y in train_dl:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            avg_loss = running_loss / total
            train_acc = correct / total

            # проста валідація на тесті
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_val, y_val in test_dl:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs_val = model(X_val)
                    loss_val = criterion(outputs_val, y_val)
                    val_loss += loss_val.item() * X_val.size(0)
                    _, preds_val = torch.max(outputs_val, 1)
                    val_correct += (preds_val == y_val).sum().item()
                    val_total += y_val.size(0)

            avg_val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            line = (
                f"Epoch {epoch}: "
                f"train_loss={avg_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}\n"
            )
            print(line.strip())
            log_file.write(line)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_acc": train_acc,
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                }
            )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), model_path)

        total_time = time.time() - start_time
        log_file.write(f"Total training time: {total_time:.2f} seconds\n")

    metrics = {
        "epochs": epochs,
        "best_val_loss": best_loss,
        "last_val_acc": history[-1]["val_acc"],
        "total_time_sec": total_time,
        "history": history,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training finished.")
    print("Best model saved to:", model_path)
    print("Metrics saved to:", metrics_path)
    print("Log saved to:", log_path)


if __name__ == "__main__":
    E = int(os.getenv("EPOCHS", "2"))
    train(epochs=E)
