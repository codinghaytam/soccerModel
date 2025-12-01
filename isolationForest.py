import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from joblib import dump





# Adjustable variables
KEYPOINTS_JSON_PATH = os.path.join("data", "keypoints.json")
OUTPUT_MODELS_DIR = os.path.join("models", "isolation_forest")
SUMMARY_PATH = os.path.join("models", "isolation_forest", "if_accuracy_summary.json")
RANDOM_SEED = 42
# Train/test/eval split (on frames). Must sum to 1.0
SPLIT_RATIOS = {
    "train": 0.7,
    "test": 0.2,
    "eval": 0.1,
}
# IsolationForest parameters (shared across models)
IF_PARAMS = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": "auto",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
# Only use x and y (ignore z)
USE_XY_ONLY = True
# Minimum frames per label to train
MIN_FRAMES_PER_LABEL = 50
# Diagram generation toggle and path
GENERATE_DIAGRAM = True
DIAGRAM_PATH = os.path.join("models", "isolation_forest", "summary_diagram.png")


def _ensure_dirs():
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)


def _load_keypoints() -> List[dict]:
    with open(KEYPOINTS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept both list of samples and object with "samples"
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    else:
        raise ValueError("keypoints.json must be a list of samples or an object with 'samples'")
    # Basic validation
    for s in samples:
        if "label" not in s or "keypoints" not in s:
            raise ValueError("Each sample must have 'label' and 'keypoints'")
    return samples


def _frame_to_feature(sample: dict) -> np.ndarray:
    kps = sample["keypoints"]
    # Expect a list of joints with x,y,(z)
    if USE_XY_ONLY:
        feat = []
        for j in kps:
            if not ("x" in j and "y" in j):
                raise ValueError("Keypoint entries must have x and y")
            feat.extend([j["x"], j["y"]])
        return np.asarray(feat, dtype=np.float64)
    else:
        feat = []
        for j in kps:
            feat.extend([j.get("x", 0.0), j.get("y", 0.0), j.get("z", 0.0)])
        return np.asarray(feat, dtype=np.float64)


def _split_indices(n: int, ratios: Dict[str, float]) -> Tuple[List[int], List[int], List[int]]:
    idxs = list(range(n))
    random.Random(RANDOM_SEED).shuffle(idxs)
    n_train = int(ratios["train"] * n)
    n_test = int(ratios["test"] * n)
    train = idxs[:n_train]
    test = idxs[n_train:n_train + n_test]
    eval_ = idxs[n_train + n_test:]
    return train, test, eval_


def _train_if_per_label(X: np.ndarray) -> IsolationForest:
    model = IsolationForest(**IF_PARAMS)
    model.fit(X)
    return model


def _train_if_per_joint(X: np.ndarray, n_joints: int) -> List[IsolationForest]:
    # X shape: [N_frames, n_joints*2] (xy)
    joint_models = []
    for j in range(n_joints):
        col_slice = slice(j * 2, j * 2 + 2)
        Xm = X[:, col_slice]
        m = IsolationForest(**IF_PARAMS)
        m.fit(Xm)
        joint_models.append(m)
    return joint_models


def _predict_joint_anomalies(joint_models: List[IsolationForest], x_frame: np.ndarray) -> List[Dict]:
    anomalies = []
    for j, m in enumerate(joint_models):
        sl = slice(j * 2, j * 2 + 2)
        xf = x_frame[sl].reshape(1, -1)
        pred = int(m.predict(xf)[0])  # -1 anomaly, 1 normal
        score = float(m.score_samples(xf)[0])
        anomalies.append({
            "joint_index": j,
            "is_anomaly": pred == -1,
            "anomaly_score": score,
        })
    return anomalies



def main():
    _ensure_dirs()
    samples = _load_keypoints()

    # Group frames by label
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)

    summary = {}

    for label, frames in by_label.items():
        if len(frames) < MIN_FRAMES_PER_LABEL:
            print(f"Skipping '{label}' (only {len(frames)} frames, need >= {MIN_FRAMES_PER_LABEL})")
            continue

        X = np.vstack([_frame_to_feature(f) for f in frames])
        n_frames, d = X.shape
        if USE_XY_ONLY:
            if d % 2 != 0:
                raise ValueError("Feature length not divisible by 2 for xy")
            n_joints = d // 2
        else:
            raise NotImplementedError("Joint count inference for xyz not implemented")

        train_idx, test_idx, eval_idx = _split_indices(n_frames, SPLIT_RATIOS)
        X_train, X_test, X_eval = X[train_idx], X[test_idx], X[eval_idx]

        # Train frame-level IF
        frame_if = _train_if_per_label(X_train)
        # Train joint-level IFs for anomaly explanation
        joint_ifs = _train_if_per_joint(X_train, n_joints)

        # Save models
        label_dir = os.path.join(OUTPUT_MODELS_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        dump(frame_if, os.path.join(label_dir, "frame_if.joblib"))
        dump(joint_ifs, os.path.join(label_dir, "joint_ifs.joblib"))

        # Evaluate
        def eval_split(X_split: np.ndarray, name: str):
            y_pred = frame_if.predict(X_split)  # -1 anomaly, 1 normal
            scores = frame_if.score_samples(X_split)
            normal_rate = float(np.mean(y_pred == 1))
            mean_score = float(np.mean(scores))
            return {
                "num_samples": int(len(X_split)),
                "normal_rate": normal_rate,
                "mean_score": mean_score,
            }

        split_metrics = {
            "train": eval_split(X_train, "train"),
            "test": eval_split(X_test, "test"),
            "eval": eval_split(X_eval, "eval"),
        }

        # Example inference output for the eval split (first 5 frames)
        examples = []
        for i in range(min(5, len(X_eval))):
            xf = X_eval[i]
            frame_score = float(frame_if.score_samples(xf.reshape(1, -1))[0])
            frame_pred = int(frame_if.predict(xf.reshape(1, -1))[0])
            joint_anoms = _predict_joint_anomalies(joint_ifs, xf)
            examples.append({
                "frame_index_in_eval": i,
                "frame_pred": frame_pred,
                "frame_score": frame_score,
                "joint_anomalies": joint_anoms,
            })

        summary[label] = {
            "frames": n_frames,
            "n_joints": n_joints,
            "metrics": split_metrics,
            "examples": examples,
        }

    # Save summary
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print short resume
    print("IsolationForest models summary:")
    for label, info in summary.items():
        m = info["metrics"]
        print(f"- {label}: train normal_rate={m['train']['normal_rate']:.3f}, test normal_rate={m['test']['normal_rate']:.3f}, eval normal_rate={m['eval']['normal_rate']:.3f}")
        print(f"  mean_scores: train={m['train']['mean_score']:.4f}, test={m['test']['mean_score']:.4f}, eval={m['eval']['mean_score']:.4f}")

    if GENERATE_DIAGRAM:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
        if plt is None:
            print("matplotlib not available, skipping diagram")
        else:
            labels = []
            eval_normal = []
            eval_score = []
            for label, info in summary.items():
                labels.append(label)
                eval_normal.append(info["metrics"]["eval"]["normal_rate"])
                eval_score.append(info["metrics"]["eval"]["mean_score"])
            if labels:
                fig, ax1 = plt.subplots(figsize=(10, 5))
                x = np.arange(len(labels))
                width = 0.35
                ax1.bar(x - width/2, eval_normal, width, label='eval_normal_rate')
                ax1.bar(x + width/2, eval_score, width, label='eval_mean_score')
                ax1.set_xticks(x)
                ax1.set_xticklabels(labels, rotation=45, ha='right')
                ax1.set_ylabel('Value')
                ax1.set_title('IsolationForest Eval Metrics per Label')
                ax1.legend()
                plt.tight_layout()
                plt.savefig(DIAGRAM_PATH, dpi=150)
                plt.close(fig)
                print(f"Saved diagram to {DIAGRAM_PATH}")
            else:
                print("No labels to plot")


if __name__ == "__main__":
    main()
