"""PyTorch neural network for patient risk classification."""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


class CardioNet(nn.Module):
    """Feed-forward network with batch normalisation and dropout."""

    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super(CardioNet, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),

            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, 1)
            # No sigmoid — BCEWithLogitsLoss applies it internally
        )

        self._init_weights()

    def _init_weights(self):
        """Apply Xavier uniform initialisation to linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class EarlyStopper:
    """Stop training when validation AUC plateaus. Saves best model state."""

    def __init__(self, patience: int = 20, min_delta: float = 0.002):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_auc  = 0.0
        self.best_state = None

    def check(self, val_auc: float, model_state: dict) -> bool:
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc   = val_auc
            self.best_state = {k: v.clone() for k, v in model_state.items()}
            self.counter    = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train(X_train, y_train, X_val, y_val,
          epochs=200, lr=0.001, batch_size=32,
          dropout=0.3, on_epoch_end=None):
    """
    Train CardioNet with AdamW, cosine LR schedule, and early stopping.

    Args:
        on_epoch_end: optional callback(epoch, total, train_loss, val_loss, val_auc)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_tensor(X, y=None, is_float=True):
        Xt = torch.FloatTensor(X).to(device)
        if y is not None:
            yt = torch.FloatTensor(
                y.values if hasattr(y, "values") else np.asarray(y, dtype=np.float32)
            ).unsqueeze(1).to(device)
            return Xt, yt
        return Xt

    X_tr, y_tr = to_tensor(X_train, y_train)
    X_v,  y_v  = to_tensor(X_val, y_val)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True   # avoids BatchNorm crash when last batch has size 1
    )

    model     = CardioNet(input_dim=X_train.shape[1], dropout_rate=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    stopper   = EarlyStopper(patience=20)

    history = {k: [] for k in ["train_loss", "val_loss", "val_auc", "train_auc"]}

    for epoch in range(1, epochs + 1):

        # ── training ──────────────────────────────────────────────
        model.train()
        epoch_loss  = 0.0
        all_probs   = []
        all_labels  = []

        for Xb, yb in loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(yb.cpu().numpy().flatten())

        # ── validation ────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss   = criterion(val_logits, y_v).item()
            val_probs  = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_true   = y_v.cpu().numpy().flatten()

        avg_train_loss = epoch_loss / len(loader)
        val_auc   = roc_auc_score(val_true, val_probs)
        train_auc = roc_auc_score(all_labels, all_probs)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["train_auc"].append(train_auc)

        scheduler.step()

        if on_epoch_end and epoch % 5 == 0:
            on_epoch_end(epoch, epochs, avg_train_loss, val_loss, val_auc)

        should_stop = stopper.check(val_auc, model.state_dict())
        if should_stop:
            break

    # restore best weights
    if stopper.best_state:
        model.load_state_dict(stopper.best_state)

    _save(model, X_train.shape[1])
    return model, history, device, stopper.best_auc


def predict_proba(X: np.ndarray) -> np.ndarray:
    """Load saved model and return probability of class 1."""
    model, device = _load()
    model.eval()
    with torch.no_grad():
        X_t    = torch.FloatTensor(X).to(device)
        logits = model(X_t)
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs


def evaluate_saved(X_test, y_test) -> dict:
    """Runs full evaluation on the saved DNN."""
    from sklearn.metrics import roc_curve, confusion_matrix, classification_report
    y_true = y_test if isinstance(y_test, np.ndarray) else np.asarray(y_test)
    probs  = predict_proba(X_test)
    preds  = (probs >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_true, probs)
    return {
        "name":             "CardioNet (PyTorch DNN)",
        "accuracy":         round(accuracy_score(y_true, preds), 4),
        "roc_auc":          round(roc_auc_score(y_true, probs), 4),
        "f1":               round(f1_score(y_true, preds), 4),
        "optimal_threshold":0.5,
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "class_report":     classification_report(y_true, preds, output_dict=True),
        "fpr":              fpr.tolist(),
        "tpr":              tpr.tolist(),
    }


def _save(model, input_dim):
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cardionet.pth")
    with open("models/cardionet_meta.pkl", "wb") as f:
        pickle.dump({"input_dim": input_dim}, f)


def _load():
    with open("models/cardionet_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CardioNet(input_dim=meta["input_dim"]).to(device)
    model.load_state_dict(torch.load("models/cardionet.pth", map_location=device))
    return model, device
