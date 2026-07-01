import os
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed — SHAP plots will be skipped")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE


def normalize_label(val):
    """Map raw dataset label → clean display name using LABEL_MAP."""
    return LABEL_MAP.get(str(val).strip().lower(), str(val).strip())


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
# ── FIX 1: Raised to 2 GB — handles full Kaggle CSVs ──────────────────────────
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024   # 2 GB

# ── FIX 2: Increased sample size + chunk-based reading (see /analyze) ─────────
SAMPLE_SIZE = 200_000   # rows used for ML; raised from 50 000
CHUNK_SIZE  = 100_000   # rows per chunk when streaming large files

# Maps raw dataset labels → clean display names
LABEL_MAP = {
    "benign": "Normal",        "normal": "Normal",
    "0": "Normal",
    "bot": "Botnet/C2",        "botnet": "Botnet/C2",
    "infilteration": "Infiltration", "infiltration": "Infiltration",
    "portscan": "Recon",       "reconnaissance": "Recon",
    "ddos": "DDoS",
    "dos hulk": "DoS",         "dos goldeneye": "DoS",
    "dos slowloris": "DoS",    "dos slowhttptest": "DoS",
    "dos": "DoS",
    "web attack \x96 brute force": "Web Attack",
    "web attack \x96 xss": "Web Attack",
    "web attack \x96 sql injection": "Web Attack",
    "brute force": "Web Attack",
    "ftp-patator": "Credential Attack",
    "ssh-patator": "Credential Attack",
    "heartbleed": "Exploit",
    "apt": "APT",
    "lateral movement": "APT",
    "exfiltration": "APT",
}

SEVERITY = {
    "Normal": 0, "Recon": 1, "Credential Attack": 2,
    "Web Attack": 2, "DoS": 3, "DDoS": 3,
    "Infiltration": 4, "Botnet/C2": 4, "Exploit": 4, "APT": 5,
}

PALETTE = {
    "Normal": "#1D9E75",  "Recon": "#EF9F27",
    "Credential Attack": "#D85A30", "Web Attack": "#D85A30",
    "DoS": "#E24B4A",     "DDoS": "#E24B4A",
    "Infiltration": "#993C1D", "Botnet/C2": "#7F77DD",
    "Exploit": "#D4537E", "APT": "#E24B4A",
}


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#0D1117", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")


def read_large_csv(file_obj):
    """
    FIX 3: Chunked CSV reader — streams the file in CHUNK_SIZE-row blocks
    instead of loading everything into RAM at once, then reservoir-samples
    down to SAMPLE_SIZE rows.  This keeps memory flat regardless of file size.

    Returns (df_sample, was_sampled, original_rows).
    """
    chunks = []
    total_rows = 0
    reader = pd.read_csv(file_obj, low_memory=False, chunksize=CHUNK_SIZE)

    for chunk in reader:
        total_rows += len(chunk)
        chunks.append(chunk)

        # Early-exit once we've buffered enough rows to sample from
        if total_rows >= SAMPLE_SIZE * 3:
            # Drain remaining chunks just to count rows, without storing them
            for remaining in reader:
                total_rows += len(remaining)
            break

    df = pd.concat(chunks, ignore_index=True)

    was_sampled = False
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        was_sampled = True

    return df, was_sampled, total_rows


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # ── Step A: Receive and read the file ─────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file"}), 400

    was_sampled = False
    original_rows = 0

    try:
        # FIX 3 applied here — replaces the old single pd.read_csv call
        df, was_sampled, original_rows = read_large_csv(f)
    except Exception as e:
        return jsonify({
            "was_sampled":   was_sampled,
            "original_rows": original_rows,
            "sample_size":   SAMPLE_SIZE,
            "error": f"Could not read CSV: {e}"
        }), 400

    df.columns = df.columns.str.strip()

    # ── Step B: Find the label column ─────────────────────────
    label_col = None
    for candidate in ["Label", "label", "Class", "class",
                      "Attack", "attack", "Category", "category"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        return jsonify({"error": "No label column found. Expected: Label, Class, Attack, or Category"}), 400

    df["_label"] = df[label_col].apply(normalize_label)

    # ── Step C: Select numeric feature columns ─────────────────
    drop_cols = {label_col, "_label", "Flow ID", "Source IP",
                 "Destination IP", "Src IP", "Dst IP", "Timestamp"}
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) < 3:
        return jsonify({"error": "Not enough numeric feature columns (need at least 3)"}), 400

    # ── Step D: Clean the data ─────────────────────────────────
    df_clean = (
        df[feature_cols + ["_label"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(df_clean) < 50:
        return jsonify({"error": "Too few valid rows after cleaning (need at least 50)"}), 400

    X = df_clean[feature_cols].values
    y_raw = df_clean["_label"].values

    # ── Step E: Encode labels ──────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_.tolist()

    # ── Step F: Apply SMOTE ────────────────────────────────────
    counts = pd.Series(y_raw).value_counts()
    min_count = counts.min()
    k = min(5, int(min_count) - 1)
    smote_applied = False

    if k >= 1 and len(counts) > 1:
        try:
            sm = SMOTE(k_neighbors=k, random_state=42)
            X, y = sm.fit_resample(X, y)
            smote_applied = True
        except Exception:
            pass

    # ── Step G: Scale features ─────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Step H: Train/test split ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # ── Step I: Train Random Forest ────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

    # ── Step J: Evaluate ───────────────────────────────────────
    report = classification_report(
        y_test, y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    accuracy = round(report["accuracy"] * 100, 2)

    try:
        if len(classes) == 2:
            auc = round(roc_auc_score(y_test, y_prob[:, 1]) * 100, 2)
        else:
            auc = round(roc_auc_score(y_test, y_prob, multi_class="ovr") * 100, 2)
    except Exception:
        auc = None

    # ── Step K: Per-class metrics ──────────────────────────────
    class_metrics = []
    for cls in classes:
        r = report.get(cls, {})
        class_metrics.append({
            "name":      cls,
            "precision": round(r.get("precision", 0) * 100, 1),
            "recall":    round(r.get("recall",    0) * 100, 1),
            "f1":        round(r.get("f1-score",  0) * 100, 1),
            "support":   int(r.get("support", 0)),
            "color":     PALETTE.get(cls, "#8B949E"),
            "severity":  SEVERITY.get(cls, 0),
        })

    # ── Step L: Feature importances ────────────────────────────
    importances = rf.feature_importances_
    top_idx = importances.argsort()[::-1][:15]
    top_features = [
        {"name": feature_cols[i], "importance": round(float(importances[i]) * 100, 2)}
        for i in top_idx
    ]

    # ── Step M: Per-row predictions ────────────────────────────
    X_orig = scaler.transform(
        df_clean[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .values
    )
    preds_orig = le.inverse_transform(rf.predict(X_orig))
    probs_orig = rf.predict_proba(X_orig).max(axis=1)
    actual_orig = df_clean["_label"].values

    rows = []
    for i in range(min(500, len(df_clean))):
        pred = preds_orig[i]
        rows.append({
            "row":        i + 1,
            "actual":     actual_orig[i],
            "predicted":  pred,
            "confidence": round(float(probs_orig[i]) * 100, 1),
            "severity":   SEVERITY.get(pred, 0),
            "color":      PALETTE.get(pred, "#8B949E"),
            "correct":    actual_orig[i] == pred,
        })

    # ── Step N: Class distribution ─────────────────────────────
    dist = df_clean["_label"].value_counts().to_dict()
    dist_data = [
        {"label": k, "count": int(v), "color": PALETTE.get(k, "#8B949E")}
        for k, v in dist.items()
    ]

    # ── SHAP explainability ─────────────────────────────────────
    shap_summary_img   = None
    shap_waterfall_img = None

    if SHAP_AVAILABLE:
        try:
            bg_sample = shap.sample(X_train, min(200, len(X_train)), random_state=42)
            explainer = shap.TreeExplainer(rf, bg_sample)
            shap_sample = X_test[:min(100, len(X_test))]
            shap_values = explainer.shap_values(shap_sample)

            if isinstance(shap_values, list):
                mean_shap = np.abs(np.array(shap_values)).mean(axis=0)
            else:
                mean_shap = np.abs(shap_values)

            fig1 = plt.figure(figsize=(9, 5), facecolor="#0D1117")
            shap.summary_plot(mean_shap, shap_sample, feature_names=feature_cols,
                              plot_type="bar", show=False, max_display=12, color="#7F77DD")
            ax1 = plt.gca()
            ax1.set_facecolor("#161B22")
            fig1.patch.set_facecolor("#0D1117")
            ax1.tick_params(colors="#8B949E", labelsize=8)
            for spine in ax1.spines.values():
                spine.set_edgecolor("#30363D")
            ax1.grid(color="#21262D", linewidth=0.5, axis="x")
            plt.title("SHAP Feature Importance (mean |SHAP value|)",
                      color="#E6EDF3", fontsize=11, pad=10)
            shap_summary_img = fig_to_base64(fig1)

            severity_scores = {i: SEVERITY.get(c, 0) for i, c in enumerate(classes)}
            target_cls_idx  = max(severity_scores, key=severity_scores.get)

            if isinstance(shap_values, list):
                sv_single = shap_values[target_cls_idx][0]
                exp_val   = explainer.expected_value[target_cls_idx]
            else:
                sv_single = shap_values[0]
                exp_val   = explainer.expected_value

            fig2 = plt.figure(figsize=(10, 4), facecolor="#0D1117")
            shap.waterfall_plot(
                shap.Explanation(
                    values        = sv_single,
                    base_values   = float(exp_val),
                    data          = shap_sample[0],
                    feature_names = feature_cols
                ),
                max_display=10, show=False
            )
            ax2 = plt.gca()
            ax2.set_facecolor("#161B22")
            fig2.patch.set_facecolor("#0D1117")
            ax2.tick_params(colors="#8B949E", labelsize=8)
            plt.title(f"SHAP Waterfall — sample prediction ({classes[target_cls_idx]})",
                      color="#E6EDF3", fontsize=10, pad=8)
            shap_waterfall_img = fig_to_base64(fig2)

        except Exception as e:
            print(f"SHAP error (non-fatal): {e}")
            shap_summary_img   = None
            shap_waterfall_img = None

    # ── Step O: Return to browser ──────────────────────────────
    return jsonify({
        "accuracy":           accuracy,
        "auc":                auc,
        "smote_applied":      smote_applied,
        "total_rows":         len(df_clean),
        "original_rows":      original_rows,        # FIX 4: always returned
        "was_sampled":        was_sampled,
        "sample_size":        SAMPLE_SIZE,
        "feature_count":      len(feature_cols),
        "class_count":        len(classes),
        "classes":            classes,
        "class_metrics":      class_metrics,
        "top_features":       top_features,
        "dist_data":          dist_data,
        "rows":               rows,
        "shap_summary_img":   shap_summary_img,
        "shap_waterfall_img": shap_waterfall_img,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)