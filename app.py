import os
import warnings
warnings.filterwarnings("ignore")          # Suppress sklearn/SMOTE warnings in terminal

import shap
import matplotlib
matplotlib.use("Agg")   # IMPORTANT: use non-GUI backend — no screen needed
import matplotlib.pyplot as plt
import io               # for saving plots to memory
import base64           # for converting images to text the browser can receive
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
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Maps raw dataset labels → clean display names
# Works with CICIDS 2017/2018, UNSW-NB15, CTU-13, NSL-KDD
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

# Severity levels — used to colour-code the results table
SEVERITY = {
    "Normal": 0, "Recon": 1, "Credential Attack": 2,
    "Web Attack": 2, "DoS": 3, "DDoS": 3,
    "Infiltration": 4, "Botnet/C2": 4, "Exploit": 4, "APT": 5,
}

# Colours for each class — sent to the frontend for chart colours
PALETTE = {
    "Normal": "#1D9E75",  "Recon": "#EF9F27",
    "Credential Attack": "#D85A30", "Web Attack": "#D85A30",
    "DoS": "#E24B4A",     "DDoS": "#E24B4A",
    "Infiltration": "#993C1D", "Botnet/C2": "#7F77DD",
    "Exploit": "#D4537E", "APT": "#E24B4A",
}
def fig_to_base64(fig):
    """Save a matplotlib figure to memory and return it as a base64 string."""
    buf = io.BytesIO()             # create an in-memory file buffer
    fig.savefig(
        buf,
        format="png",
        dpi=130,                   # high resolution for the paper
        bbox_inches="tight",       # don't cut off axis labels
        facecolor="#0D1117",       # dark background matches your CSS
        edgecolor="none"
    )
    buf.seek(0)                    # rewind buffer to the beginning
    plt.close(fig)                 # free memory — important in a web server
    return base64.b64encode(buf.read()).decode("utf-8")
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

    try:
        df = pd.read_csv(f, low_memory=False)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    df.columns = df.columns.str.strip()    # Remove any accidental spaces in column names

    # ── Step B: Find the label column ─────────────────────────
    label_col = None
    for candidate in ["Label", "label", "Class", "class",
                      "Attack", "attack", "Category", "category"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        return jsonify({"error": "No label column found. Expected: Label, Class, Attack, or Category"}), 400

    # Normalise all labels using our map
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

    X = df_clean[feature_cols].values      # Feature matrix (numbers only)
    y_raw = df_clean["_label"].values      # Label array (strings)

    # ── Step E: Encode labels to integers ─────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_raw)            # e.g. BENIGN→0, Bot→1, DoS→2
    classes = le.classes_.tolist()         # ["BENIGN", "Bot", "DoS", ...]

    # ── Step F: Apply SMOTE ────────────────────────────────────
    counts = pd.Series(y_raw).value_counts()
    min_count = counts.min()

    # k_neighbors must be less than the smallest class size
    k = min(5, int(min_count) - 1)
    smote_applied = False

    if k >= 1 and len(counts) > 1:
        try:
            sm = SMOTE(k_neighbors=k, random_state=42)
            X, y = sm.fit_resample(X, y)
            smote_applied = True
        except Exception:
            pass                           # If SMOTE fails, continue without it

    # ── Step G: Scale features ─────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Step H: Train/test split ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.25,
        random_state=42,
        stratify=y           # Ensures same class ratio in both splits
    )

    # ── Step I: Train Random Forest ────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=150,    # 150 decision trees in the forest
        random_state=42,     # Fixed seed → reproducible results
        n_jobs=-1            # Use all CPU cores → faster training
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

    # ── Step J: Evaluate ───────────────────────────────────────
    report = classification_report(
        y_test, y_pred,
        target_names=classes,
        output_dict=True,    # Returns a dict instead of a string
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

    # ── Step K: Build per-class metrics for frontend charts ────
    class_metrics = []
    for cls in classes:
        r = report.get(cls, {})
        class_metrics.append({
            "name": cls,
            "precision": round(r.get("precision", 0) * 100, 1),
            "recall":    round(r.get("recall", 0) * 100, 1),
            "f1":        round(r.get("f1-score", 0) * 100, 1),
            "support":   int(r.get("support", 0)),
            "color":     PALETTE.get(cls, "#8B949E"),
            "severity":  SEVERITY.get(cls, 0),
        })

    # ── Step L: Feature importances ────────────────────────────
    importances = rf.feature_importances_
    top_idx = importances.argsort()[::-1][:15]   # Top 15 features
    top_features = [
        {
            "name": feature_cols[i],
            "importance": round(float(importances[i]) * 100, 2)
        }
        for i in top_idx
    ]

    # ── Step M: Per-row predictions on original data ───────────
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
    for i in range(min(500, len(df_clean))):    # First 500 rows max
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
    # ── SHAP explainability ──────────────────────────────────────
    shap_summary_img  = None
    shap_waterfall_img = None

    try:
        # Use a background sample for speed (200 rows is enough)
        bg_sample = shap.sample(X_train, min(200, len(X_train)), random_state=42)

        # TreeExplainer is fast and exact for Random Forest
        explainer = shap.TreeExplainer(rf, bg_sample)

        # Compute SHAP values on up to 100 test rows
        shap_sample = X_test[:min(100, len(X_test))]
        shap_values = explainer.shap_values(shap_sample)

        # ── Plot 1: Summary bar chart ─────────────────────────────
        # shap_values is a list (one array per class) for multi-class
        # We take the mean absolute value across all classes
        if isinstance(shap_values, list):
            mean_shap = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values)

        fig1 = plt.figure(figsize=(9, 5), facecolor="#0D1117")
        shap.summary_plot(
            mean_shap,
            shap_sample,
            feature_names=feature_cols,
            plot_type="bar",
            show=False,             # don't try to open a window
            max_display=12,         # show top 12 features
            color="#7F77DD"         # purple bars
        )
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

        # ── Plot 2: Waterfall for one prediction ─────────────────
        # Pick the highest-severity class to explain
        severity_scores = {i: SEVERITY.get(c, 0) for i, c in enumerate(classes)}
        target_cls_idx  = max(severity_scores, key=severity_scores.get)

        # Get SHAP values and expected value for that class
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
            max_display = 10,       # top 10 features in waterfall
            show        = False
        )
        ax2 = plt.gca()
        ax2.set_facecolor("#161B22")
        fig2.patch.set_facecolor("#0D1117")
        ax2.tick_params(colors="#8B949E", labelsize=8)
        plt.title(
            f"SHAP Waterfall — sample prediction ({classes[target_cls_idx]})",
            color="#E6EDF3", fontsize=10, pad=8
        )
        shap_waterfall_img = fig_to_base64(fig2)

    except Exception as e:
        # SHAP failing should NOT crash the whole app
        print(f"SHAP error (non-fatal): {e}")
        shap_summary_img   = None
        shap_waterfall_img = None

    # ── Step O: Return everything to the browser ───────────────
    return jsonify({
        "accuracy":          accuracy,
        "auc":               auc,
        "smote_applied":     smote_applied,
        "total_rows":        len(df_clean),
        "feature_count":     len(feature_cols),
        "class_count":       len(classes),
        "classes":           classes,
        "class_metrics":     class_metrics,
        "top_features":      top_features,
        "dist_data":         dist_data,
        "rows":              rows,
        "shap_summary_img":  shap_summary_img,   # NEW
        "shap_waterfall_img": shap_waterfall_img, # NEW
    })


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, port=5000)
    import shap
import matplotlib
matplotlib.use("Agg")   # IMPORTANT: use non-GUI backend — no screen needed
import matplotlib.pyplot as plt
import io               # for saving plots to memory
import base64           # for converting images to text the browser can receive
