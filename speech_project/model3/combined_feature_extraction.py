"""
Urban Sound Classification using PyTorch CNN (Combined Features)
================================================================
Features used per audio file:
  1. MFCCs (40 bands)
  2. Log-Mel Spectrogram (128 bands)
  3. Spectral Contrast (7 bands)
  Total stacked features = 175 bands along the Y-axis.

Classifies audio clips from the UrbanSound8K dataset into 10 categories.

Directory structure expected:
  ./UrbanSound8K/UrbanSound8K/
      metadata/UrbanSound8K.csv
      audio/fold1/ ... fold10/
"""

import os
import numpy as np
import pandas as pd
import librosa
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────

DATASET_ROOT = "./UrbanSound8K/UrbanSound8K"
METADATA_CSV = os.path.join(DATASET_ROOT, "metadata", "UrbanSound8K.csv")
AUDIO_ROOT   = os.path.join(DATASET_ROOT, "audio")

# Feature Hyperparameters
MAX_PAD_LEN  = 174          # fixed time-axis length (zero-padded)
N_MFCC       = 40           # number of MFCC coefficients
N_MELS       = 128          # number of Mel bands
# Spectral contrast returns 7 bands by default (6 bands + 1 valley)
# Total height of our feature image will be: 40 + 128 + 7 = 175

NUM_EPOCHS   = 72
BATCH_SIZE   = 256
LEARNING_RATE = 1e-3
TEST_SIZE    = 0.2
RANDOM_SEED  = 42

CHECKPOINT_PATH = "saved_models/weights_best_cnn_combined.pt"
RESULTS_DIR     = "results_combined_features"  # Dedicated folder for this run
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 2. Feature Extraction (Combined)
# ─────────────────────────────────────────────

def extract_features(file_name: str) -> np.ndarray | None:
    """
    Load an audio file, extract MFCC, Mel Spectrogram, and Spectral Contrast.
    Stack them vertically into a single image-like matrix and pad/truncate.
    Shape returned: (175, 174)
    """
    try:
        audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
        
        # 1. MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        
        # 2. Log-Mel Spectrogram
        melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=N_MELS)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        
        # 3. Spectral Contrast
        # Computes difference in amplitude between peaks and valleys in the spectrum
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        
        # Ensure the time dimension matches across all 3 features before stacking
        # (librosa usually keeps these uniform, but slight rounding differences can occur)
        min_time_steps = min(mfccs.shape[1], log_melspec.shape[1], contrast.shape[1])
        
        mfccs       = mfccs[:, :min_time_steps]
        log_melspec = log_melspec[:, :min_time_steps]
        contrast    = contrast[:, :min_time_steps]
        
        # Stack vertically to create a rich multi-feature map -> Shape: (175, min_time_steps)
        combined_features = np.vstack([mfccs, log_melspec, contrast])
        
        # Pad or truncate the time axis to MAX_PAD_LEN
        pad_width = MAX_PAD_LEN - combined_features.shape[1]
        if pad_width < 0:
            combined_features = combined_features[:, :MAX_PAD_LEN]
        else:
            combined_features = np.pad(
                combined_features, 
                pad_width=((0, 0), (0, pad_width)), 
                mode="constant"
            )
            
        return combined_features
        
    except Exception as e:
        print(f"Error parsing {file_name}: {e}")
        return None


def build_feature_dataframe(metadata_csv: str, audio_root: str) -> pd.DataFrame:
    """
    Iterate over every row in the metadata CSV, extract features,
    and return a DataFrame with columns ['feature', 'class_label'].
    """
    metadata = pd.read_csv(metadata_csv)
    features = []

    for _, row in metadata.iterrows():
        file_name = os.path.join(
            audio_root,
            f"fold{row['fold']}",
            str(row["slice_file_name"]),
        )
        data = extract_features(file_name)
        if data is not None:
            # Note: Checking for both 'class' and 'class_name' depending on the CSV version
            label = row["class"] if "class" in row else row["class_name"]
            features.append([data, label])

    df = pd.DataFrame(features, columns=["feature", "class_label"])
    print(f"Finished feature extraction from {len(df)} files.")
    return df


# ─────────────────────────────────────────────
# 3. PyTorch Dataset
# ─────────────────────────────────────────────

class UrbanSoundDataset(Dataset):
    """Wraps numpy multi-feature arrays and integer labels for DataLoader consumption."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X shape: (N, 175, 174)  →  add channel dim  →  (N, 1, 175, 174)
        self.X = torch.tensor(X[:, np.newaxis, :, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# 4. CNN Model
# ─────────────────────────────────────────────

class UrbanSoundCNN(nn.Module):
    """
    Four-block Conv2D CNN. 
    Because we use AdaptiveAvgPool2d, the model handles the larger 
    Y-dimension (175 vs 40) automatically without breaking the Linear layer.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.block1 = self._conv_block(1,   16)
        self.block2 = self._conv_block(16,  32)
        self.block3 = self._conv_block(32,  64)
        self.block4 = self._conv_block(64, 128)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)   # Squeezes spatial dimensions to 1x1
        self.classifier      = nn.Linear(128, num_classes)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)   # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 128)
        return self.classifier(x)     # (B, num_classes)  — raw logits


# ─────────────────────────────────────────────
# 5. Training
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * len(y_batch)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += len(y_batch)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        total_loss    += loss.item() * len(y_batch)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += len(y_batch)

    return total_loss / total_samples, total_correct / total_samples


def train(model, train_loader, val_loader, num_epochs: int):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    best_val_acc = 0.0
    start = datetime.now()
    
    # Track history for plotting
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%"
        )

        # Save checkpoint when validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ↳ Checkpoint saved (val_acc={val_acc*100:.2f}%)")

    duration = datetime.now() - start
    print(f"\nTraining completed in: {duration}")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    
    return history


# ─────────────────────────────────────────────
# 6. Inference
# ─────────────────────────────────────────────

@torch.no_grad()
def predict(model: nn.Module, file_name: str, le: LabelEncoder) -> str:
    """Return the predicted class label for a single audio file."""
    features = extract_features(file_name)
    if features is None:
        return "Error: could not extract features."

    # shape: (1, 1, 175, 174)
    tensor = torch.tensor(features[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
    model.eval()
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_idx   = int(np.argmax(probs))
    predicted_class = le.inverse_transform([predicted_idx])[0]

    print(f"\nPredicted class: {predicted_class}\n")
    print("Per-class probabilities:")
    for i, p in enumerate(probs):
        category = le.inverse_transform([i])[0]
        print(f"  {category:<25}: {p:.6f}")

    return predicted_class


@torch.no_grad()
def get_all_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) arrays over an entire DataLoader."""
    model.eval()
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        logits  = model(X_batch)
        preds   = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_preds)


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────

def main():
    print(f"Using device: {DEVICE}\n")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Feature extraction ──────────────────
    print("Extracting Combined Features (MFCC + Mel + Spectral Contrast)…")
    print("This requires heavy computation. Please be patient.")
    features_df = build_feature_dataframe(METADATA_CSV, AUDIO_ROOT)

    X = np.array(features_df["feature"].tolist())           # (N, 175, 174)
    y_raw = np.array(features_df["class_label"].tolist())   # string labels

    # ── Label encoding ──────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)                            # integer labels
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {list(le.classes_)}\n")

    # ── Train/test split ────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    train_dataset = UrbanSoundDataset(X_train, y_train)
    test_dataset  = UrbanSoundDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Build model ─────────────────────────
    model = UrbanSoundCNN(num_classes=num_classes).to(DEVICE)
    print(model)

    criterion = nn.CrossEntropyLoss()
    _, pre_train_acc = evaluate(model, test_loader, criterion)
    print(f"\nPre-training accuracy: {pre_train_acc*100:.4f}%\n")

    # ── Train ───────────────────────────────
    history = train(model, train_loader, test_loader, num_epochs=NUM_EPOCHS)

    # ── Plot Training Curves ────────────────
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([acc * 100 for acc in history['train_acc']], label='Train Accuracy')
    plt.plot([acc * 100 for acc in history['val_acc']], label='Validation Accuracy', linestyle='--')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'))
    plt.close()

    # ── Load best checkpoint & final eval ───
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    _, train_acc = evaluate(model, train_loader, criterion)
    _, test_acc  = evaluate(model, test_loader,  criterion)
    print(f"\nFinal Training Accuracy : {train_acc*100:.4f}%")
    print(f"Final Testing  Accuracy : {test_acc*100:.4f}%")

    # ── Evaluation & File Saving ────────────
    y_true, y_pred = get_all_predictions(model, test_loader)
    
    report = classification_report(y_true, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # Save metrics to text file
    with open(os.path.join(RESULTS_DIR, "results.txt"), "w") as f:
        f.write("URBAN SOUND CLASSIFICATION - COMBINED FEATURES (MFCC + MEL + CONTRAST)\n")
        f.write("="*70 + "\n")
        f.write(f"Final Training Accuracy : {train_acc*100:.4f}%\n")
        f.write(f"Final Testing  Accuracy : {test_acc*100:.4f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()

    print(f"\n✅ Results and plots have been successfully saved to the '{RESULTS_DIR}' folder.")


if __name__ == "__main__":
    main()