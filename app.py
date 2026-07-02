import os
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

SAMPLE_SIZE  = 15_000
RF_TREES     = 50
SHAP_BG      = 30
SHAP_EXPLAIN = 20

LABEL_MAP = {
    "benign":"Normal","normal":"Normal","0":"Normal",
    "bot":"Botnet/C2","botnet":"Botnet/C2",
    "infilteration":"Infiltration","infiltration":"Infiltration",
    "portscan":"Recon","reconnaissance":"Recon","ddos":"DDoS",
    "dos hulk":"DoS","dos goldeneye":"DoS","dos slowloris":"DoS",
    "dos slowhttptest":"DoS","dos":"DoS",
    "web attack \x96 brute force":"Web Attack","web attack – brute force":"Web Attack",
    "web attack \x96 xss":"Web Attack","web attack – xss":"Web Attack",
    "web attack \x96 sql injection":"Web Attack","web attack – sql injection":"Web Attack",
    "brute force":"Web Attack","ftp-patator":"Credential Attack",
    "ssh-patator":"Credential Attack","heartbleed":"Exploit",
    "apt":"APT","lateral movement":"APT","exfiltration":"APT",
    "credential attack":"Credential Attack","web attack":"Web Attack",
}
SEVERITY = {
    "Normal":0,"Recon":1,"Credential Attack":2,"Web Attack":2,
    "DoS":3,"DDoS":3,"Infiltration":4,"Botnet/C2":4,"Exploit":4,"APT":5,
}
PALETTE = {
    "Normal":"#1D9E75","Recon":"#EF9F27","Credential Attack":"#D85A30",
    "Web Attack":"#D85A30","DoS":"#E24B4A","DDoS":"#E24B4A",
    "Infiltration":"#993C1D","Botnet/C2":"#7F77DD","Exploit":"#D4537E","APT":"#E24B4A",
}

def normalize_label(val):
    return LABEL_MAP.get(str(val).strip().lower(), str(val).strip())

def detect_label_column(df):
    """Find the label column — strip names, try candidates, then heuristic."""
    # Always strip whitespace from column names first
    df.columns = [c.strip() for c in df.columns]

    candidates = [
        "label","class","attack","category","type","target",
        "attack_cat","attack_type","traffic_category","classification",
        "y","outcome","result","subcategory","threat",
    ]
    col_lower = {c.lower(): c for c in df.columns}

    # Pass 1: exact name match
    for cand in candidates:
        if cand in col_lower:
            return col_lower[cand]

    # Pass 2: find any string/object column with 2–100 unique values
    keywords = {
        "benign","normal","attack","dos","ddos","bot","recon","apt",
        "scan","probe","exploit","heartbleed","infiltration","backdoor",
        "shellcode","worm","flood","credential","lateral","exfil",
        "malicious","anomaly","intrusion","portscan",
    }
    for col in df.columns:
        if df[col].dtype != object:
            continue
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) < 2 or len(unique_vals) > 100:
            continue
        vals_lower = [str(v).lower() for v in unique_vals]
        if any(any(kw in v for kw in keywords) for v in vals_lower):
            return col

    # Pass 3: last column fallback (many IDS datasets put label last)
    return df.columns[-1]


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="#0D1117", edgecolor="none")
    buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file"}), 400

    # ── Read CSV — simple, no chunking, no stratified sampling ────────────────
    # Chunked groupby was silently dropping the Label column in pandas 2.x
    try:
        df = pd.read_csv(f, low_memory=False)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    original_rows = len(df)
    was_sampled   = False

    # Strip column names immediately after reading
    df.columns = [c.strip() for c in df.columns]

    # Sample BEFORE any processing if too large
    if len(df) > SAMPLE_SIZE:
        df          = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        was_sampled = True

    # ── Find label column ──────────────────────────────────────────────────────
    label_col = detect_label_column(df)

    # Validate the detected column actually looks like labels
    if df[label_col].dtype != object and df[label_col].nunique() > 50:
        all_cols = ", ".join(df.columns.tolist())
        return jsonify({
            "error": (
                f"Cannot find a label/class column. "
                f"All columns: [{all_cols}]. "
                f"Please rename your label column to 'Label'."
            )
        }), 400

    df["_label"] = df[label_col].apply(normalize_label)

    # ── Features ───────────────────────────────────────────────────────────────
    drop_cols = {label_col, "_label", "Flow ID", "Source IP", "Destination IP",
                 "Src IP", "Dst IP", "Timestamp", "src_ip", "dst_ip",
                 "flow_id", "timestamp"}
    feature_cols = [c for c in df.columns
                    if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    if len(feature_cols) < 3:
        return jsonify({"error": "Not enough numeric feature columns (need at least 3)"}), 400

    df_clean = df[feature_cols + ["_label"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_clean) < 50:
        return jsonify({"error": "Too few valid rows after cleaning"}), 400

    X     = df_clean[feature_cols].values
    y_raw = df_clean["_label"].values
    le    = LabelEncoder(); y = le.fit_transform(y_raw); classes = le.classes_.tolist()

    # ── SMOTE ──────────────────────────────────────────────────────────────────
    counts = pd.Series(y_raw).value_counts(); k = min(5, int(counts.min()) - 1)
    smote_applied = False
    if k >= 1 and len(counts) > 1:
        try:
            X, y = SMOTE(k_neighbors=k, random_state=42).fit_resample(X, y)
            smote_applied = True
        except Exception:
            pass

    # ── Train ──────────────────────────────────────────────────────────────────
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test); y_prob = rf.predict_proba(X_test)

    # ── Metrics ────────────────────────────────────────────────────────────────
    report   = classification_report(y_test, y_pred, target_names=classes,
                                     output_dict=True, zero_division=0)
    accuracy = round(report["accuracy"] * 100, 2)
    try:
        auc = round(roc_auc_score(y_test, y_prob,
                    multi_class="ovr" if len(classes) > 2 else "raise") * 100, 2) \
              if len(classes) > 2 else \
              round(roc_auc_score(y_test, y_prob[:, 1]) * 100, 2)
    except Exception:
        auc = None

    class_metrics = [{"name": cls,
        "precision": round(report.get(cls,{}).get("precision",0)*100, 1),
        "recall":    round(report.get(cls,{}).get("recall",   0)*100, 1),
        "f1":        round(report.get(cls,{}).get("f1-score", 0)*100, 1),
        "support":   int(report.get(cls,{}).get("support",0)),
        "color":     PALETTE.get(cls,"#8B949E"),
        "severity":  SEVERITY.get(cls,0)} for cls in classes]

    top_idx      = rf.feature_importances_.argsort()[::-1][:15]
    top_features = [{"name": feature_cols[i],
                     "importance": round(float(rf.feature_importances_[i])*100, 2)}
                    for i in top_idx]

    X_orig = scaler.transform(
        df_clean[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(0).values)
    preds_orig  = le.inverse_transform(rf.predict(X_orig))
    probs_orig  = rf.predict_proba(X_orig).max(axis=1)
    actual_orig = df_clean["_label"].values
    rows = [{"row":i+1,"actual":actual_orig[i],"predicted":preds_orig[i],
             "confidence":round(float(probs_orig[i])*100,1),
             "severity":SEVERITY.get(preds_orig[i],0),
             "color":PALETTE.get(preds_orig[i],"#8B949E"),
             "correct":actual_orig[i]==preds_orig[i]}
            for i in range(min(500,len(df_clean)))]

    dist_data = [{"label":k,"count":int(v),"color":PALETTE.get(k,"#8B949E")}
                 for k,v in df_clean["_label"].value_counts().to_dict().items()]

    # ── SHAP ───────────────────────────────────────────────────────────────────
    shap_summary_img = shap_waterfall_img = None
    if SHAP_AVAILABLE:
        try:
            bg  = shap.sample(X_train, min(SHAP_BG,len(X_train)), random_state=42)
            exp = shap.TreeExplainer(rf, bg)
            ss  = X_test[:min(SHAP_EXPLAIN,len(X_test))]
            sv  = exp.shap_values(ss)
            mean_shap = np.abs(np.array(sv)).mean(axis=0) if isinstance(sv,list) else np.abs(sv)

            fig1 = plt.figure(figsize=(9,5), facecolor="#0D1117")
            shap.summary_plot(mean_shap, ss, feature_names=feature_cols,
                              plot_type="bar", show=False, max_display=12, color="#7F77DD")
            ax1=plt.gca(); ax1.set_facecolor("#161B22"); fig1.patch.set_facecolor("#0D1117")
            ax1.tick_params(colors="#8B949E",labelsize=8)
            for sp in ax1.spines.values(): sp.set_edgecolor("#30363D")
            ax1.grid(color="#21262D",linewidth=0.5,axis="x")
            plt.title("SHAP Feature Importance",color="#E6EDF3",fontsize=11,pad=10)
            shap_summary_img = fig_to_base64(fig1)

            tgt = max(range(len(classes)), key=lambda i: SEVERITY.get(classes[i],0))
            sv1 = sv[tgt][0] if isinstance(sv,list) else sv[0]
            ev  = exp.expected_value[tgt] if isinstance(sv,list) else exp.expected_value

            fig2 = plt.figure(figsize=(10,4), facecolor="#0D1117")
            shap.waterfall_plot(shap.Explanation(values=sv1, base_values=float(ev),
                                data=ss[0], feature_names=feature_cols),
                                max_display=10, show=False)
            ax2=plt.gca(); ax2.set_facecolor("#161B22"); fig2.patch.set_facecolor("#0D1117")
            ax2.tick_params(colors="#8B949E",labelsize=8)
            plt.title(f"SHAP Waterfall ({classes[tgt]})",color="#E6EDF3",fontsize=10,pad=8)
            shap_waterfall_img = fig_to_base64(fig2)
        except Exception as e:
            print(f"SHAP error: {e}")

    return jsonify({
        "accuracy":accuracy,"auc":auc,"smote_applied":smote_applied,
        "total_rows":len(df_clean),"original_rows":original_rows,
        "was_sampled":was_sampled,"sample_size":SAMPLE_SIZE,
        "feature_count":len(feature_cols),"class_count":len(classes),
        "classes":classes,"class_metrics":class_metrics,
        "top_features":top_features,"dist_data":dist_data,"rows":rows,
        "shap_summary_img":shap_summary_img,"shap_waterfall_img":shap_waterfall_img,
        "label_col_detected":label_col,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)