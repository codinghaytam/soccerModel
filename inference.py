
from __future__ import annotations
import os
import json
import argparse
from typing import List, Dict
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
GMM_DIR = os.path.join(MODELS_DIR, "gmm")
# Default video path defined in the file (override with --video). Update this path as needed.
VIDEO_PATH = os.path.join(DATA_DIR, 'C:\\Users\\ULTRA PC\\PycharmProjects\\PythonProject1\\data\\foul\\foul_0001.avi')  # TODO: replace with your actual video filename

# ---------------- Placeholder keypoint extraction -----------------

def extract_keypoints_from_video(video_path: str) -> np.ndarray:

    vid_id = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(DATA_DIR, "keypoints.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"keypoints.json not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError("keypoints.json is empty")
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    # Filter records for this video id
    frames = [r for r in records if str(r.get("video")) == vid_id]
    if not frames:
        raise ValueError(f"No keypoint frames found for video id '{vid_id}' derived from '{video_path}'")
    frames_sorted = sorted(frames, key=lambda r: r.get("frame", 0))
    coords_list = []
    for fr in frames_sorted:
        kps = fr.get("keypoints", [])
        if not kps:
            continue
        arr = np.array([[kp.get("x", 0.0), kp.get("y", 0.0), kp.get("z", 0.0)] for kp in kps], dtype=np.float32)
        coords_list.append(arr)
    if not coords_list:
        raise ValueError(f"Frames found for '{vid_id}' but all empty keypoints.")
    coords = np.stack(coords_list, axis=0)  # (T, J, 3)
    return coords

# --------------- Autoencoder definition (same as training) ---------------
import torch.nn as nn

class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_joints: int):
        super().__init__()
        self.A = nn.Parameter(torch.eye(num_joints))
        self.theta = nn.Linear(in_channels, out_channels)
    def forward(self, x):
        N, T, J, C = x.shape
        A = torch.softmax(self.A, dim=-1)
        x = x.view(N * T, J, C)
        x = torch.matmul(A, x)
        x = self.theta(x)
        x = x.view(N, T, J, -1)
        return x

class SkeletonGCNAutoencoder(nn.Module):
    def __init__(self, num_joints: int, in_channels: int = 3, hidden: int = 64, latent: int = 32):
        super().__init__()
        # Match training model layer names to load weights correctly
        self.encoder_g1 = GraphConv(in_channels, hidden, num_joints)
        self.encoder_g2 = GraphConv(hidden, latent, num_joints)
        self.decoder_g1 = GraphConv(latent, hidden, num_joints)
        self.decoder_g2 = GraphConv(hidden, in_channels, num_joints)
        self.relu = nn.ReLU()
    def forward(self, x, mask=None):
        # Mirror training forward pass
        h = self.relu(self.encoder_g1(x))
        z = self.encoder_g2(h)
        h_dec = self.relu(self.decoder_g1(z))
        out = self.decoder_g2(h_dec)
        if mask is not None:
            out = out * mask.unsqueeze(-1).unsqueeze(-1)
        return out

# --------------- Utility loaders ---------------

def load_autoencoder(label: str, num_joints: int, device: torch.device) -> SkeletonGCNAutoencoder:
    weight_path = os.path.join(MODELS_DIR, label, "autoencoder.pt")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Autoencoder weights not found for label '{label}': {weight_path}")
    model = SkeletonGCNAutoencoder(num_joints=num_joints).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# --- GMM load (adapted from gmm_model.py) ---
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

def load_gmm(label: str) -> GaussianMixture:
    path = os.path.join(GMM_DIR, f"{label}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"GMM file not found for label '{label}': {path}")
    data = np.load(path, allow_pickle=True)
    cov_type = "full"
    if "covariance_type" in data:
        cov_type_arr = data["covariance_type"]
        try:
            cov_type = str(cov_type_arr.item()) if hasattr(cov_type_arr, "item") else str(cov_type_arr)
        except Exception:
            cov_type = str(cov_type_arr)
    if all(k in data.files for k in ("weights", "means", "covariances")):
        weights = data["weights"].astype(np.float64)
        means = data["means"].astype(np.float64)
        covs = data["covariances"].astype(np.float64)
    else:
        arr_keys = sorted([k for k in data.files if k.startswith("arr_")], key=lambda k: int(k.split("_")[1]))
        if len(arr_keys) < 3:
            raise KeyError(f"Unexpected npz format for {path}: {data.files}")
        weights = data[arr_keys[0]].astype(np.float64)
        means = data[arr_keys[1]].astype(np.float64)
        covs = data[arr_keys[2]].astype(np.float64)
    gmm = GaussianMixture(n_components=weights.shape[0], covariance_type=cov_type)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covs
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covs, cov_type)
    return gmm

def load_scaler():
    scaler_path = os.path.join(GMM_DIR, "scaler.npz")
    if not os.path.exists(scaler_path):
        return None
    sc = np.load(scaler_path)
    return sc["mean_"], sc["scale_"]

# --------------- Feature extraction (match training) ---------------

def extract_features(coords: np.ndarray, center_joint: int | None = None) -> np.ndarray:
    T, J, C = coords.shape
    x = coords.copy()
    if center_joint is not None and 0 <= center_joint < J:
        ref = x[:, center_joint:center_joint+1, :]
        x = x - ref
    pos_mean = x.mean(axis=0)
    pos_std = x.std(axis=0) + 1e-6
    vel = np.diff(x, axis=0) if T > 1 else np.zeros_like(x)
    vel_mean = vel.mean(axis=0)
    vel_std = vel.std(axis=0) + 1e-6
    traj_len = np.linalg.norm(vel, axis=2).sum(axis=0)
    feat = np.concatenate([
        pos_mean.flatten(), pos_std.flatten(), vel_mean.flatten(), vel_std.flatten(), traj_len.flatten()
    ], axis=0)
    return feat.astype(np.float64)

# --------------- Inference routines ---------------

def autoencoder_errors(coords: np.ndarray, label: str) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_joints = coords.shape[1]
    model = load_autoencoder(label, num_joints, device)
    x = torch.from_numpy(coords).unsqueeze(0).to(device)  # (1,T,J,3)
    mask = torch.ones((1, x.shape[1]), dtype=torch.bool, device=device)
    with torch.no_grad():
        recon = model(x, mask=mask)
        diff = (recon - x) * mask.unsqueeze(-1).unsqueeze(-1)
        per_joint_frame_l2 = torch.sqrt((diff ** 2).sum(dim=-1)).squeeze(0).detach().cpu().numpy()  # (T,J)

    # Basic aggregates (backward compatible)
    mean_error = float(per_joint_frame_l2.mean())
    per_frame_avg = per_joint_frame_l2.mean(axis=1)
    per_joint_avg = per_joint_frame_l2.mean(axis=0)

    # Detailed anomaly summaries
    T, J = per_joint_frame_l2.shape
    joint_summaries = []
    for j in range(J):
        v = per_joint_frame_l2[:, j]
        mu = float(v.mean())
        sd = float(v.std())
        mx = float(v.max())
        mx_f = int(v.argmax())
        p95 = float(np.percentile(v, 95))
        # Dynamic threshold = max(mean+2*std, 95th percentile)
        thr = float(max(mu + 2.0 * sd, p95))
        an_idx = np.where(v > thr)[0]
        # Top-5 worst frames for this joint
        top_idx = np.argsort(v)[-5:][::-1]
        top_frames = [{"frame": int(f), "error": float(v[f])} for f in top_idx]
        joint_summaries.append({
            "joint": j,
            "mean": mu,
            "std": sd,
            "p95": p95,
            "threshold": thr,
            "max_error": mx,
            "max_error_frame": mx_f,
            "num_anomalous_frames": int(an_idx.size),
            "anomalous_frames": [int(f) for f in an_idx.tolist()[:20]],  # cap list size
            "top_frames": top_frames,
        })

    # Frame-wise view: top-5 most erroneous frames overall and their worst joints
    top_frames_idx = np.argsort(per_frame_avg)[-5:][::-1]
    frames_detail = []
    for f in top_frames_idx:
        row = per_joint_frame_l2[f, :]
        worst_j_idx = np.argsort(row)[-3:][::-1]
        frames_detail.append({
            "frame": int(f),
            "mean_error": float(per_frame_avg[f]),
            "worst_joints": [{"joint": int(j), "error": float(row[j])} for j in worst_j_idx],
        })

    overall_std = float(per_joint_frame_l2.std())
    overall_thr = float(mean_error + 2.0 * overall_std)
    frames_over_thr = np.where(per_frame_avg > overall_thr)[0].tolist()

    return {
        "mean_error": mean_error,
        "errors_shape": [int(T), int(J)],
        "per_frame_avg": per_frame_avg.tolist(),
        "per_joint_avg": per_joint_avg.tolist(),
        "detail": {
            "overall": {
                "mean": mean_error,
                "std": overall_std,
                "threshold": overall_thr,
                "num_frames_over_threshold": int(len(frames_over_thr)),
                "frames_over_threshold": [int(f) for f in frames_over_thr[:50]],
            },
            "top_frames_overall": frames_detail,
            "joints": joint_summaries,
            "notes": "Thresholds are heuristic (max(mean+2*std, 95th percentile)). Errors are L2 over (x,y,z).",
        },
    }

def gmm_classify(coords: np.ndarray, center_joint: int | None = None) -> Dict:
    feat = extract_features(coords, center_joint=center_joint)
    scaler = load_scaler()
    if scaler is not None:
        mean_, scale_ = scaler
        feat = (feat - mean_) / np.where(scale_ == 0, 1.0, scale_)
    # Load all gmms
    labels = [f.split('.')[0] for f in os.listdir(GMM_DIR) if f.endswith('.npz') and f != 'scaler.npz']
    if not labels:
        raise FileNotFoundError("No GMM models found.")
    scores = {}
    for lbl in labels:
        gmm = load_gmm(lbl)
        scores[lbl] = float(gmm.score_samples(feat.reshape(1, -1))[0])
    # Higher log-likelihood => better match
    sorted_labels = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "scores": scores,
        "predicted_label": sorted_labels[0][0],
        "top3": sorted_labels[:3],
    }

def autoencoder_best(coords: np.ndarray) -> Dict:
    labels = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d)) and os.path.exists(os.path.join(MODELS_DIR, d, 'autoencoder.pt'))]
    if not labels:
        raise FileNotFoundError("No autoencoder models found.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_joints = coords.shape[1]
    x = torch.from_numpy(coords).unsqueeze(0).to(device)
    mask = torch.ones((1, x.shape[1]), dtype=torch.bool, device=device)
    errs = {}
    for lbl in labels:
        model = load_autoencoder(lbl, num_joints, device)
        recon = model(x, mask=mask)
        diff = (recon - x) * mask.unsqueeze(-1).unsqueeze(-1)
        mse = (diff ** 2).mean().item()
        errs[lbl] = mse
    best = sorted(errs.items(), key=lambda kv: kv[1])[0]
    return {"errors": errs, "predicted_label": best[0], "best_mse": best[1]}

def passToLLM(prompt: str) -> str:
    model_name = "deepseek-ai/deepseek-llm-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    messages = [
        {"role": "user", "content": "prompt"}
    ]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=10000)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    return result
# --------------- Main CLI ---------------

def main():
    ap = argparse.ArgumentParser()
    # Make video optional; use VIDEO_PATH if not provided.
    ap.add_argument("--video", help="Path to input video. If omitted, uses VIDEO_PATH constant inside the script.")
    ap.add_argument("--label", default="corner",help="Known label (move type). If provided, compute error under that label.")
    ap.add_argument("--center_joint", type=int, default=None, help="Optional center joint index for feature extraction (GMM).")
    args = ap.parse_args()

    video_path = args.video if args.video else VIDEO_PATH

    try:
        coords = extract_keypoints_from_video(video_path)
    except NotImplementedError:
        print(json.dumps({"error": "extract_keypoints_from_video not implemented", "video_path": video_path}))
        return

    if coords.ndim != 3 or coords.shape[2] < 2:
        print(json.dumps({"error": f"Unexpected coords shape {coords.shape}", "video_path": video_path}))
        return

    result = {"video": video_path, "T": coords.shape[0], "J": coords.shape[1]}

    if args.label:
        # Error analysis for provided label
        try:
            ae = autoencoder_errors(coords, args.label)
            result["autoencoder"] = ae
        except Exception as e:
            result["autoencoder_error"] = str(e)
        # GMM likelihood & anomaly scoring vs others
        try:
            gmm_res = gmm_classify(coords, center_joint=args.center_joint)
            result["gmm"] = gmm_res
            if args.label in gmm_res["scores"]:
                # anomaly score: difference between self score and best score
                best_score = max(gmm_res["scores"].values())
                self_score = gmm_res["scores"][args.label]
                result["gmm_anomaly_score"] = float(best_score - self_score)
        except Exception as e:
            result["gmm_error"] = str(e)
    else:
        # Perform classification
        try:
            gmm_res = gmm_classify(coords, center_joint=args.center_joint)
            result["gmm_classification"] = gmm_res
        except Exception as e:
            result["gmm_error"] = str(e)
        try:
            ae_res = autoencoder_best(coords)
            result["autoencoder_classification"] = ae_res
        except Exception as e:
            result["autoencoder_error"] = str(e)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
