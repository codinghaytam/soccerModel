"""Gaussian Mixture Model (GMM) classification on 3D skeleton sequences.

This script trains one GMM per label using fixed-length feature vectors
extracted from variable-length skeleton sequences. It then evaluates
classification accuracy by choosing the label with the highest log-likelihood.

Dependencies: numpy, torch (optional for loading), scikit-learn.
"""
import os
import json
from typing import List, Dict, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Reuse data paths from the project structure
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
JSON_PATH = os.path.join(DATA_DIR, "keypoints.json")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
GMM_DIR = os.path.join(MODELS_DIR, "gmm")
os.makedirs(GMM_DIR, exist_ok=True)


# ------------------------- Data loading -------------------------

def _read_records(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_sequences() -> List[Tuple[str, str, np.ndarray]]:
    """Load sequences grouped by video.

    Returns list of tuples: (sample_id, label, coords) where coords shape is (T, J, 3).
    """
    records = _read_records(JSON_PATH)
    by_video: Dict[str, List[dict]] = {}
    for r in records:
        vid = str(r["video"])
        by_video.setdefault(vid, []).append(r)
    out: List[Tuple[str, str, np.ndarray]] = []
    for vid, frames in by_video.items():
        frames_sorted = sorted(frames, key=lambda x: x["frame"])
        label = str(frames_sorted[0].get("label", "unknown"))
        coords_list = []
        for fr in frames_sorted:
            kps = fr["keypoints"]
            arr = np.array([[kp.get("x", 0.0), kp.get("y", 0.0), kp.get("z", 0.0)] for kp in kps], dtype=np.float32)
            coords_list.append(arr)
        coords = np.stack(coords_list, axis=0)  # (T, J, 3)
        out.append((vid, label, coords))
    return out


# ------------------------- Feature extraction -------------------------

def extract_features(coords: np.ndarray, center_joint: int | None = None) -> np.ndarray:
    """Extract a fixed-length feature vector from (T, J, 3).

    Heuristics used:
    - Optional centering by a root joint (e.g., pelvis) to remove global translation.
    - Compute per-joint mean and std over time (xyz): shape (J, 3) x 2.
    - Compute velocity stats: mean and std of first temporal differences.
    - Compute overall trajectory length (sum of velocities).

    Returns a 1D feature vector.
    """
    T, J, C = coords.shape
    x = coords.copy()
    if center_joint is not None and 0 <= center_joint < J:
        ref = x[:, center_joint:center_joint+1, :]  # (T, 1, 3)
        x = x - ref
    # Per-joint position stats
    pos_mean = x.mean(axis=0)           # (J, 3)
    pos_std = x.std(axis=0) + 1e-6      # (J, 3)
    # Velocities
    vel = np.diff(x, axis=0)            # (T-1, J, 3)
    if vel.shape[0] == 0:
        vel = np.zeros_like(x)
    vel_mean = vel.mean(axis=0)         # (J, 3)
    vel_std = vel.std(axis=0) + 1e-6    # (J, 3)
    # Trajectory length per joint
    traj_len = np.linalg.norm(vel, axis=2).sum(axis=0)  # (J,)

    feat = np.concatenate([
        pos_mean.flatten(),
        pos_std.flatten(),
        vel_mean.flatten(),
        vel_std.flatten(),
        traj_len.flatten(),
    ], axis=0)
    return feat.astype(np.float32)


def build_dataset(center_joint: int | None = None) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """Build feature matrix X and label vector y.

    Returns:
    - X: shape (N, F)
    - y: shape (N,), string labels
    - sample_ids: list of video ids
    - indices_per_label: dict mapping label -> list of indices
    """
    sequences = load_sequences()
    sample_ids: List[str] = []
    labels: List[str] = []
    feats: List[np.ndarray] = []
    indices_per_label: Dict[str, List[int]] = {}

    for i, (vid, label, coords) in enumerate(sequences):
        f = extract_features(coords, center_joint=center_joint)
        sample_ids.append(vid)
        labels.append(label)
        feats.append(f)
        indices_per_label.setdefault(label, []).append(i)

    X = np.stack(feats, axis=0)
    y = np.array(labels)
    # Cast to float64 for numerical stability
    X = X.astype(np.float64)
    return X, y, sample_ids, indices_per_label


# ------------------------- GMM training and evaluation -------------------------

def train_gmm_per_label(n_components: int = 2, covariance_type: str = "full", center_joint: int | None = None, reg_covar: float = 1e-6) -> Dict[str, str]:
    """Train one GMM per label and save them under models/gmm/<label>.npz.

    Applies StandardScaler to features globally, uses float64, and adds reg_covar for stability.
    If a label has too few samples, automatically reduces n_components.
    """
    X, y, _, indices = build_dataset(center_joint=center_joint)
    labels = sorted(indices.keys())

    # Standardize features globally (fit on all X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Save global scaler for inference
    scaler_path = os.path.join(GMM_DIR, "scaler.npz")
    np.savez(scaler_path, mean_=scaler.mean_.astype(np.float64), scale_=scaler.scale_.astype(np.float64))

    saved_paths: Dict[str, str] = {}
    for label in labels:
        idx = indices[label]
        X_label = X_scaled[idx]
        n_samples = X_label.shape[0]
        # Adjust components if samples are few
        k = min(n_components, max(1, n_samples))
        if k > n_samples:
            k = n_samples
        # Initialize and fit with stability settings
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42, reg_covar=reg_covar)
        try:
            gmm.fit(X_label)
        except Exception as e:
            # Fallbacks: increase reg_covar, reduce components
            print(f"GMM fit failed for '{label}' with {k} components: {e}. Retrying with reg_covar=1e-4...")
            try:
                gmm = GaussianMixture(n_components=max(1, k // 2), covariance_type=covariance_type, random_state=42, reg_covar=1e-4)
                gmm.fit(X_label)
            except Exception as e2:
                print(f"Second attempt failed for '{label}': {e2}. Using 1 component and higher reg_covar=1e-3...")
                gmm = GaussianMixture(n_components=1, covariance_type=covariance_type, random_state=42, reg_covar=1e-3)
                gmm.fit(X_label)
        # Save parameters
        path = os.path.join(GMM_DIR, f"{label}.npz")
        np.savez(path,
                 weights=gmm.weights_.astype(np.float64),
                 means=gmm.means_.astype(np.float64),
                 covariances=gmm.covariances_.astype(np.float64),
                 covariance_type=covariance_type)
        saved_paths[label] = path
        print(f"Saved GMM for '{label}' to {path} | X_label: {X_label.shape} | components: {gmm.n_components}")
    return saved_paths


def _load_gmm(path: str) -> GaussianMixture:
    data = np.load(path, allow_pickle=True)
    # Determine covariance_type value robustly
    cov_type = "full"
    if "covariance_type" in data:
        cov_type_arr = data["covariance_type"]
        try:
            cov_type = str(cov_type_arr.item()) if hasattr(cov_type_arr, "item") else str(cov_type_arr)
        except Exception:
            cov_type = str(cov_type_arr)

    # Retrieve weights, means, covariances with fallbacks
    if all(k in data.files for k in ("weights", "means", "covariances")):
        weights = np.asarray(data["weights"], dtype=np.float64)
        means = np.asarray(data["means"], dtype=np.float64)
        covs = np.asarray(data["covariances"], dtype=np.float64)
    else:
        # Fallback to positional arrays arr_0, arr_1, arr_2
        arr_keys = [k for k in data.files if k.startswith("arr_")]
        if len(arr_keys) < 3:
            raise KeyError(f"Expected keys 'weights','means','covariances' or positional 'arr_0..2' in {path}, found: {data.files}")
        # Sort by index
        arr_keys_sorted = sorted(arr_keys, key=lambda k: int(k.split("_")[1]))
        weights = np.asarray(data[arr_keys_sorted[0]], dtype=np.float64)
        means = np.asarray(data[arr_keys_sorted[1]], dtype=np.float64)
        covs = np.asarray(data[arr_keys_sorted[2]], dtype=np.float64)

    n_components = int(weights.shape[0])
    gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covs

    # Derive precisions cholesky
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, cov_type)
    return gmm


def evaluate_gmm_accuracy(center_joint: int | None = None) -> Dict[str, object]:
    """Evaluate classification accuracy using per-label GMMs.

    For each sample feature f, compute log-likelihood under each label's GMM,
    pick the label with highest score.
    """
    X, y, sample_ids, _ = build_dataset(center_joint=center_joint)

    # Load scaler if available
    scaler_path = os.path.join(GMM_DIR, "scaler.npz")
    if os.path.exists(scaler_path):
        sc = np.load(scaler_path)
        mean_ = sc["mean_"]
        scale_ = sc["scale_"]
        X = (X - mean_) / np.where(scale_ == 0, 1.0, scale_)
    else:
        print("Warning: Scaler not found. Ensure consistent preprocessing.")

    # Load all available GMMs
    gmm_paths = [os.path.join(GMM_DIR, f) for f in os.listdir(GMM_DIR) if f.endswith(".npz") and f != "scaler.npz"]
    if not gmm_paths:
        raise FileNotFoundError("No GMM model files found under models/gmm/. Train them first.")
    label_names = [os.path.splitext(os.path.basename(p))[0] for p in gmm_paths]
    gmms = {label: _load_gmm(path) for label, path in zip(label_names, gmm_paths)}

    # Compute scores
    scores = np.stack([gmms[label].score_samples(X) for label in label_names], axis=1)  # (N, L)
    pred_idx = scores.argmax(axis=1)
    pred_labels = np.array([label_names[i] for i in pred_idx])

    # Accuracy
    correct = (pred_labels == y)
    overall_acc = float(correct.mean())
    # Per-label
    labels_unique = sorted(set(y.tolist()))
    per_label_acc = {}
    confusion = {l_true: {l_pred: 0 for l_pred in label_names} for l_true in labels_unique}

    for i in range(len(y)):
        confusion[y[i]][pred_labels[i]] += 1

    for l in labels_unique:
        mask = (y == l)
        if mask.any():
            per_label_acc[l] = float((pred_labels[mask] == y[mask]).mean())
        else:
            per_label_acc[l] = 0.0

    summary = {
        "overall_accuracy": overall_acc,
        "per_label_accuracy": per_label_acc,
        "labels": label_names,
        "confusion_matrix": confusion,
        "num_samples_evaluated": int(len(y)),
    }

    # Save
    out_path = os.path.join(GMM_DIR, "gmm_accuracy_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved GMM accuracy summary to {out_path}")
    print("Overall accuracy:", overall_acc)
    for l, acc in per_label_acc.items():
        print(f"  {l}: {acc:.4f}")

    return summary


if __name__ == "__main__":
    # Example usage:
    # 1) Train GMMs per label
    train_gmm_per_label(n_components=2, covariance_type="full", center_joint=None)
    # 2) Evaluate accuracy
    evaluate_gmm_accuracy(center_joint=None)
