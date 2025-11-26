"""GCN training on 3D skeleton sequences.

This script expects a keypoints.csv or keypoints.json file in data/ with rows like:
    sample_id, label, frame, joint, x, y, z

Goal:
- Build one graph convolutional model per label (action / player type / etc.), or
  a single multi-class model if preferred.
- For a given sequence, the model predicts the "ideal" movement and we measure
  the error (deviation) per joint and per frame.

You will likely need to adapt the data loading part to match your exact format.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "keypoints.csv")
JSON_PATH = os.path.join(DATA_DIR, "keypoints.json")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ------------------------- Data utilities -------------------------

@dataclass
class SkeletonSample:
    sample_id: str
    label: str
    coords: np.ndarray  # (T, J, 3)


def load_keypoints() -> List[SkeletonSample]:
    """Load skeleton data from CSV or JSON into a list of SkeletonSample.

    Adapted to the user's keypoints.json format where each record is:
        {
          "video": "corner_0001",
          "frame": 0,
          "label": "corner",
          "keypoints": [
             {"x": ..., "y": ..., "z": ...},
             ... (J joints)
          ]
        }

    The file can be either a JSON array of such records, or NDJSON
    (one JSON object per line).
    """
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Detect format: array JSON vs one-object-per-line (NDJSON)
        if text.startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]

        # Group frames by video id
        by_video: Dict[str, List[dict]] = {}
        for r in records:
            vid = str(r["video"])
            by_video.setdefault(vid, []).append(r)

        samples: List[SkeletonSample] = []
        for vid, frames in by_video.items():
            # sort by frame index
            frames_sorted = sorted(frames, key=lambda r: r["frame"])
            label = str(frames_sorted[0]["label"]) if "label" in frames_sorted[0] else "unknown"

            coords_list: List[np.ndarray] = []
            for fr in frames_sorted:
                kps = fr["keypoints"]  # list of joints
                frame_arr = np.array(
                    [[kp["x"], kp["y"], kp.get("z", 0.0)] for kp in kps],
                    dtype=np.float32,
                )  # (J, 3)
                coords_list.append(frame_arr)

            # (T, J, 3)
            coords = np.stack(coords_list, axis=0)
            samples.append(
                SkeletonSample(
                    sample_id=vid,
                    label=label,
                    coords=coords,
                )
            )

        return samples

    # Fallback stub for CSV. Implement according to your schema.
    if os.path.exists(CSV_PATH):
        raise NotImplementedError(
            "CSV loading is not implemented. Please adapt load_keypoints()."
        )

    raise FileNotFoundError("No keypoints.json or keypoints.csv found in data/.")


def pad_sequences(seqs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Pad sequences of shape (T_i, J, C) to (N, T_max, J, C) and return mask.

    Returns
    -------
    padded : np.ndarray, float32, shape (N, T_max, J, C)
    mask   : np.ndarray, bool,     shape (N, T_max)
    """
    n = len(seqs)
    max_t = max(s.shape[0] for s in seqs)
    j = seqs[0].shape[1]
    c = seqs[0].shape[2]

    padded = np.zeros((n, max_t, j, c), dtype=np.float32)
    mask = np.zeros((n, max_t), dtype=bool)
    for i, s in enumerate(seqs):
        t = s.shape[0]
        padded[i, :t] = s
        mask[i, :t] = True
    return padded, mask


class SkeletonDataset(Dataset):
    def __init__(self, samples: List[SkeletonSample], label_filter: str | None = None):
        if label_filter is not None:
            samples = [s for s in samples if s.label == label_filter]
        self.samples = samples
        self.label_filter = label_filter

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # coords: (T, J, 3)
        return {
            "coords": torch.from_numpy(s.coords),
            "sample_id": s.sample_id,
            "label": s.label,
        }


def collate_skeleton(batch):
    coords_list = [b["coords"] for b in batch]  # list of (T, J, 3)
    sample_ids = [b["sample_id"] for b in batch]
    labels = [b["label"] for b in batch]

    seqs = [c.numpy() for c in coords_list]
    padded, mask = pad_sequences(seqs)  # (N, T, J, 3), (N, T)
    # Convert to tensors: X: (N, T, J, 3)
    x = torch.from_numpy(padded)  # float32
    mask = torch.from_numpy(mask)  # bool
    return {"x": x, "mask": mask, "sample_ids": sample_ids, "labels": labels}


# ------------------------- Simple GCN building blocks -------------------------

class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        # Learnable adjacency (symmetric constrained via softplus if needed)
        self.A = nn.Parameter(torch.eye(num_joints))
        self.theta = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """x: (N, T, J, C_in)"""
        N, T, J, C = x.shape
        A = torch.softmax(self.A, dim=-1)  # (J, J)
        x = x.view(N * T, J, C)  # (N*T, J, C)
        x = torch.matmul(A, x)  # (N*T, J, C)
        x = self.theta(x)  # (N*T, J, C_out)
        x = x.view(N, T, J, -1)
        return x


class SkeletonGCNAutoencoder(nn.Module):
    """Per-label autoencoder: reconstructs the same sequence.

    High reconstruction error => abnormal / incorrect movement.
    """

    def __init__(self, num_joints: int, in_channels: int = 3, hidden: int = 64, latent: int = 32):
        super().__init__()
        self.encoder_g1 = GraphConv(in_channels, hidden, num_joints)
        self.encoder_g2 = GraphConv(hidden, latent, num_joints)
        self.decoder_g1 = GraphConv(latent, hidden, num_joints)
        self.decoder_g2 = GraphConv(hidden, in_channels, num_joints)
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        # x: (N, T, J, 3)
        h = self.relu(self.encoder_g1(x))
        z = self.encoder_g2(h)
        h_dec = self.relu(self.decoder_g1(z))
        recon = self.decoder_g2(h_dec)  # (N, T, J, 3)
        if mask is not None:
            recon = recon * mask.unsqueeze(-1).unsqueeze(-1)
        return recon


# ------------------------- Training loop per label -------------------------

def train_per_label(models_dir: str = MODELS_DIR, epochs: int = 20, batch_size: int = 8, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_samples = load_keypoints()
    labels = sorted({s.label for s in all_samples})
    print("Found labels:", labels)

    training_summary: Dict[str, Dict] = {}

    for label in labels:
        print(f"\n=== Training autoencoder for label: {label} ===")
        ds = SkeletonDataset(all_samples, label_filter=label)
        if len(ds) < 2:
            print(f"Skipping label {label} (not enough samples)")
            continue

        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_skeleton)

        # Infer joints from first sample
        sample0 = ds[0]["coords"]
        num_joints = sample0.shape[1]

        model = SkeletonGCNAutoencoder(num_joints=num_joints).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss(reduction="none")

        mean_loss = 0.0  # initialize to avoid potential unassigned reference
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_frames = 0
            for batch in loader:
                x = batch["x"].to(device)  # (N, T, J, 3)
                mask = batch["mask"].to(device)  # (N, T)

                optim.zero_grad()
                recon = model(x, mask=mask)
                # per-element loss: (N, T, J, 3)
                loss_elem = loss_fn(recon, x)
                # apply mask on time dimension
                loss_elem = loss_elem * mask.unsqueeze(-1).unsqueeze(-1)
                loss = loss_elem.sum() / mask.sum().clamp(min=1)
                loss.backward()
                optim.step()

                epoch_loss += loss.item() * mask.sum().item()
                n_frames += mask.sum().item()

            mean_loss = epoch_loss / max(n_frames, 1)
            print(f"Label {label} | Epoch {epoch}/{epochs} | mean frame loss: {mean_loss:.6f}")

        # Save model
        label_dir = os.path.join(models_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(label_dir, "autoencoder.pt"))
        training_summary[label] = {"epochs": epochs, "mean_frame_loss": float(mean_loss)}

    # After training all label models, compute classification accuracies
    try:
        acc_summary = evaluate_min_error_accuracy(models_dir=models_dir)
        per_label_acc = acc_summary.get("per_label_accuracy", {})
        overall_acc = acc_summary.get("overall_accuracy", 0.0)
        for lbl, stats in training_summary.items():
            if lbl in per_label_acc:
                stats["classification_accuracy"] = per_label_acc[lbl]
        training_summary["overall_accuracy"] = overall_acc
    except Exception as e:
        print(f"Accuracy evaluation skipped due to error: {e}")

    # Save global training summary (now includes accuracies)
    summary_path = os.path.join(models_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
    print("Training finished. Summary saved to", summary_path)


# ------------------------- Inference: error per joint & frame -------------------------

@torch.no_grad()
def evaluate_errors_for_label(label: str, samples: List[SkeletonSample] | None = None) -> Dict[str, np.ndarray]:
    """Return per-joint, per-frame error for each sample of a specific label.

    Returns a dict: sample_id -> errors array of shape (T, J)
    where value is mean L2 distance over xyz.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if samples is None:
        samples = load_keypoints()
    samples = [s for s in samples if s.label == label]
    if not samples:
        raise ValueError(f"No samples with label {label}")

    ds = SkeletonDataset(samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_skeleton)

    # infer joints
    num_joints = samples[0].coords.shape[1]
    model = SkeletonGCNAutoencoder(num_joints=num_joints).to(device)

    label_dir = os.path.join(MODELS_DIR, label)
    weight_path = os.path.join(label_dir, "autoencoder.pt")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Trained model not found for label {label} at {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    results: Dict[str, np.ndarray] = {}

    for batch in loader:
        x = batch["x"].to(device)  # (1, T, J, 3)
        mask = batch["mask"].to(device)  # (1, T)
        sample_id = batch["sample_ids"][0]

        recon = model(x, mask=mask)  # (1, T, J, 3)
        diff = recon - x  # (1, T, J, 3)
        diff = diff * mask.unsqueeze(-1).unsqueeze(-1)
        # L2 over xyz
        err = torch.sqrt((diff ** 2).sum(dim=-1))  # (1, T, J)
        err_np = err.squeeze(0).cpu().numpy()  # (T, J)
        results[sample_id] = err_np

    return results


@torch.no_grad()
def evaluate_min_error_accuracy(models_dir: str = MODELS_DIR, save_path: str | None = None) -> Dict[str, Dict]:
    """Compute accuracy per model/label by assigning each sample to the label whose
    autoencoder yields the lowest reconstruction error.

    Returns a dict with overall accuracy, per-label accuracy, and a confusion matrix.
    Saves JSON to models/accuracy_summary.json by default.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_keypoints()
    # Unique labels present in data
    data_labels = sorted({s.label for s in samples})

    # Load models available on disk for these labels
    models: Dict[str, SkeletonGCNAutoencoder] = {}
    num_joints = samples[0].coords.shape[1]
    for label in data_labels:
        weight_path = os.path.join(models_dir, label, "autoencoder.pt")
        if os.path.exists(weight_path):
            model = SkeletonGCNAutoencoder(num_joints=num_joints).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()
            models[label] = model
        else:
            print(f"Warning: No model found for label '{label}', skipping it in evaluation.")

    if not models:
        raise FileNotFoundError("No trained models found to evaluate.")

    labels_order = sorted(models.keys())
    label_to_idx = {l: i for i, l in enumerate(labels_order)}

    # Counters
    per_label_total = {l: 0 for l in labels_order}
    per_label_correct = {l: 0 for l in labels_order}
    # Confusion matrix counts
    confusion = {l_true: {l_pred: 0 for l_pred in labels_order} for l_true in labels_order}

    def sample_error(model: SkeletonGCNAutoencoder, coords_np: np.ndarray) -> float:
        """Mean masked MSE over T,J,3 for a single sample."""
        x = torch.from_numpy(coords_np).unsqueeze(0).to(device)  # (1, T, J, 3)
        T = x.shape[1]
        mask = torch.ones((1, T), dtype=torch.bool, device=device)
        recon = model(x, mask=mask)
        diff = (recon - x) * mask.unsqueeze(-1).unsqueeze(-1)
        se = (diff ** 2).sum()
        denom = mask.sum() * x.shape[2] * x.shape[3]  # T * J * 3
        return (se / denom.clamp(min=1)).item()

    # Evaluate all samples
    for s in samples:
        if s.label not in label_to_idx:
            # True label has no model; skip from accuracy but could still be classified
            print(f"Skipping sample '{s.sample_id}' with label '{s.label}' (no model).")
            continue
        per_label_total[s.label] += 1

        # Compute error against every model
        errs = []
        for label in labels_order:
            err = sample_error(models[label], s.coords)
            errs.append(err)
        pred_idx = int(np.argmin(errs))
        pred_label = labels_order[pred_idx]

        if pred_label == s.label:
            per_label_correct[s.label] += 1
        confusion[s.label][pred_label] += 1

    # Compute accuracies
    per_label_acc = {}
    total_correct = 0
    total_seen = 0
    for l in labels_order:
        tot = per_label_total[l]
        cor = per_label_correct[l]
        acc = float(cor) / float(tot) if tot > 0 else 0.0
        per_label_acc[l] = acc
        total_correct += cor
        total_seen += tot

    overall_acc = float(total_correct) / float(total_seen) if total_seen > 0 else 0.0

    summary = {
        "labels": labels_order,
        "overall_accuracy": overall_acc,
        "per_label_accuracy": per_label_acc,
        "confusion_matrix": confusion,
        "num_samples_evaluated": total_seen,
    }

    if save_path is None:
        save_path = os.path.join(models_dir, "accuracy_summary.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved accuracy summary to {save_path}")
    print("Overall accuracy:", overall_acc)
    for l in labels_order:
        print(f"  {l}: {per_label_acc[l]:.4f} (n={per_label_total[l]}, correct={per_label_correct[l]})")

    return summary


if __name__ == "__main__":
    # Example usage: train models per label
    train_per_label(epochs=150, batch_size=8, lr=1e-4)
