import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 128
STRIDE = 64

class ConfidenceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layer 1
        self.cnv1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5), # play with kernel size during train
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.MaxPool1d(2),
            #nn.Dropout(0.05) # play with during train
        )

        # conv layer 2
        self.cnv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3), # play with kernel size during train
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3) # play with during train
        )

        # third conv layer - likely too little data to use
        #self.cn3 = nn.Sequential(
        #    nn.Conv1d(32, 64, kernel_size=5), # play with kernel size during train
        #    nn.LazyBatchNorm1d(),
        #    nn.ReLU(),
        #    nn.MaxPool1d(2),
        #    nn.Dropout(0.4)
        #)

        self.global_pool = nn.AdaptiveAvgPool1d(2)

        # output layer
        self.out = nn.Sequential(
            nn.Linear(32*2, 64), # switch to (64, 64) if 3 conv layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x): #Shape of x: [batch_size, channels, time], e.g., [32, 4, 256]
        x = self.cnv1(x)
        x = self.cnv2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.out(x)
        return x    

# Training function
def train(model, train_loader, val_loader=None, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

        # Save after first epoch
        if epoch == 0:
            torch.save(model.state_dict(), "model_after_first_epoch.pth")
            print("Saved model after first epoch.")

        if val_loader:
            validate(model, val_loader, criterion, device)



def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"Validation Loss: {avg_val_loss:.4f}")

# cleans raw EEG data for training
def preprocess():
    def load_filtered(path):
        df = pd.read_csv(path, on_bad_lines='skip')
        return df[[col for col in df.columns if 'alpha' in col.lower() or 'gamma' in col.lower()]].values

    hf1 = load_filtered(r"C:\Users\James\Desktop\neurotech-hackathon-2025\train_data\eeg_bands_data_20250406_131204-highconfidence.csv")
    amb1 = load_filtered(r"C:/Users/James/Desktop/neurotech-hackathon-2025/train_data/eeg_bands_data_20250406_132151-amgiguous.csv")
    amb2 = load_filtered(r"C:/Users/James/Desktop/neurotech-hackathon-2025/train_data/eeg_bands_data_20250406_133720-ambigous-jillian.csv")

    coll = [1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0,
            0.95, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.8, 1.0, 1.0, 1.0, 1.0, 0.95, 1.0, 0.96, 0.9, 0.85,
            0.4, 0.6, 0.7, 0.9, 0.3, 0.5, 0.1, 0.3, 0.8, 0.4,
            0.4, 0.3, 0.2, 0.5, 0.6, 0.3, 0.65, 0.4, 0.2, 0.7, 0.5,
            0.6, 0.9, 0.4, 0.3, 0.7, 0.32, 0.7, 0.2, 0.8, 0.2]
    coll2 = [0.1, 0.7, 0.3, 0.2, 0.5, 0.2, 0.8, 0.2, 0.3, 0.1,
             0.7, 0.7, 0.1, 0.1, 0.6, 0.2, 0.4, 0.1, 0.6, 0.0,
             0.7, 0.8, 0.2, 0.7, 0.7, 0.0, 0.7, 0.2, 0.7, 0.8]

    X_raw = np.concatenate([amb1, hf1, amb2], axis=0)
    y_raw = np.array(coll + coll2)

    scaler = StandardScaler()
    X_raw = scaler.fit_transform(X_raw)

    X_windows = []
    y_windows = []

    for i in range(0, len(X_raw) - WINDOW_SIZE + 1, STRIDE):
        window = X_raw[i:i+WINDOW_SIZE]
        if window.shape[0] == WINDOW_SIZE and (i // STRIDE) < len(y_raw):
            X_windows.append(window.T)
            y_windows.append(y_raw[i // STRIDE])

    X_tensor = torch.tensor(np.stack(X_windows), dtype=torch.float32)
    y_tensor = torch.tensor(y_windows, dtype=torch.float32)
    return X_tensor, y_tensor, scaler   


def get_data_loaders(batch_size, X, y):
    dataset = TensorDataset(X, y)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

def predict_confidence_score(model: nn.Module, data: pd.DataFrame, scaler: StandardScaler, device: str = "cpu"):
    """
    Predict confidence scores for new EEG data using the trained CNN model.

    Args:
        model (nn.Module): Trained PyTorch model.
        data (np.ndarray): pandas DataFrame
        scaler (StandardScaler): Pre-fitted scaler from training.
        device (str): "cpu" or "cuda".

    Returns:
        List[float]: Confidence scores per window.
    """
    model.load_state_dict(torch.load('model_after_first_epoch.pth', map_location=device))
    model.to(device)
    model.eval()

    # Apply scaling
    data_scaled = scaler.transform(data)

    # Create windows
    X_windows = []
    for i in range(0, len(data_scaled) - WINDOW_SIZE, STRIDE):
        window = data_scaled[i:i+WINDOW_SIZE]
        if window.shape[0] == WINDOW_SIZE:
            X_windows.append(window.T)  # Shape: [channels, time]

    if not X_windows:
        raise ValueError("Input too short to create a single valid window.")

    # Convert to tensor
    X_tensor = torch.tensor(np.stack(X_windows), dtype=torch.float32).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(X_tensor).squeeze(1).cpu().numpy()

    return outputs.tolist()

def processRT(filePs, model, scaler):
    scores = []
    for i in filePs:
        dat = pd.read_csv(i, on_bad_lines='skip')

        # Filter alpha/gamma columns
        dat = dat[[col for col in dat.columns if 'alpha' in col.lower() or 'gamma' in col.lower()]]

        # Remove non-numeric entries (like bad strings)
        dat = dat.apply(pd.to_numeric, errors='coerce')  # Convert invalid strings to NaN
        dat = dat.dropna()  # Drop rows with NaN

        if dat.empty:
            print(f"Warning: {i} is empty after cleaning. Skipping.")
            continue

        temp = predict_confidence_score(data=dat, model=model, scaler=scaler)
        scores.append(temp)
    return scores

# Usage
if __name__ == "__main__":
    # Preprocess training data
    X, y, scaler = preprocess()
    train_loader, val_loader = get_data_loaders(batch_size=32, X=X, y=y)

    # Initialize model and device
    model = ConfidenceCNN()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train and save model after first epoch
    train(model, train_loader, val_loader, epochs=20, lr=1e-3, device=device)

    # Predict using the saved model
    test_file = r"C:\Users\James\Desktop\neurotech-hackathon-2025\train_data\testC.csv"
    scores = processRT([test_file], model, scaler)

    # Print confidence scores
    print(f"\nConfidence Scores from {test_file}:")
    for i, score in enumerate(scores[0]):
        print(f"Window {i + 1}: {score:.4f}")
