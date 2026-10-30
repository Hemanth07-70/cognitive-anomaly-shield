"""Cognitive Anomaly Shield — Streamlit dashboard."""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap

sys.path.insert(0, os.path.dirname(__file__))

from src.features import CATEGORIES, FEATURE_COLS, build_single_transaction
from src.models import scale
from src.explain import (
    shap_values_single, plain_english_reasons, FEATURE_LABELS,
)

# ── Config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cognitive Anomaly Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_PATH = os.path.join(os.path.dirname(__file__), "models", "artifact.joblib")

CATEGORY_DISPLAY = {
    "shopping_net":   "Online Shopping",
    "shopping_pos":   "In-Store Shopping",
    "misc_net":       "Online Misc.",
    "misc_pos":       "In-Store Misc.",
    "grocery_pos":    "Grocery (Store)",
    "grocery_net":    "Grocery (Online)",
    "gas_transport":  "Gas & Transport",
    "food_dining":    "Food & Dining",
    "entertainment":  "Entertainment",
    "personal_care":  "Personal Care",
    "health_fitness": "Health & Fitness",
    "travel":         "Travel",
    "kids_pets":      "Kids & Pets",
    "home":           "Home",
}


# ── Auto-train if artifact missing (Streamlit Cloud cold start) ────────────

def _run_training():
    from src.data_utils import load_data
    from src.features import FEATURE_COLS
    from src.models import (
        split_train_val, fit_scaler, scale as scale_df,
        train_lr, train_xgb, tune_threshold, evaluate, threshold_sweep,
    )
    from src.explain import get_lr_explainer, get_xgb_explainer, global_importance

    train_df, test_df = load_data(force_synthetic=True)
    train_split, val_split = split_train_val(train_df)
    X_train, y_train = train_split[FEATURE_COLS], train_split["is_fraud"]
    X_val,   y_val   = val_split[FEATURE_COLS],   val_split["is_fraud"]
    X_test,  y_test  = test_df[FEATURE_COLS],     test_df["is_fraud"]

    scaler    = fit_scaler(train_split)
    X_train_s = scale_df(X_train, scaler)
    X_val_s   = scale_df(X_val,   scaler)
    X_test_s  = scale_df(X_test,  scaler)

    lr        = train_lr(X_train_s, y_train)
    xgb_model = train_xgb(X_train_s, y_train, X_val_s, y_val)

    lr_thresh  = tune_threshold(lr,        X_val_s, y_val)
    xgb_thresh = tune_threshold(xgb_model, X_val_s, y_val)

    lr_results  = evaluate(lr,        X_test_s, y_test, lr_thresh)
    xgb_results = evaluate(xgb_model, X_test_s, y_test, xgb_thresh)

    X_shap       = X_test_s.sample(min(1000, len(X_test_s)), random_state=42)
    lr_explainer  = get_lr_explainer(lr, X_train_s.sample(min(500, len(X_train_s)), random_state=42))
    xgb_explainer = get_xgb_explainer(xgb_model)
    lr_imp  = global_importance(lr_explainer,  X_shap)
    xgb_imp = global_importance(xgb_explainer, X_shap)

    artifact = {
        "scaler": scaler, "lr": lr, "xgb": xgb_model,
        "lr_thresh": lr_thresh, "xgb_thresh": xgb_thresh,
        "lr_results":  {k: v for k, v in lr_results.items()  if k != "probs"},
        "xgb_results": {k: v for k, v in xgb_results.items() if k != "probs"},
        "lr_importance": lr_imp, "xgb_importance": xgb_imp,
        "lr_sweep":  threshold_sweep(lr,        X_test_s, y_test),
        "xgb_sweep": threshold_sweep(xgb_model, X_test_s, y_test),
        "n_test": len(test_df), "n_frauds": int(y_test.sum()),
        "fraud_rate": float(y_test.mean()),
        "lr_explainer": lr_explainer, "xgb_explainer": xgb_explainer,
        "X_test_sample": X_test_s.head(500).reset_index(drop=True),
        "y_test_sample": y_test.head(500).reset_index(drop=True),
    }
    os.makedirs(os.path.dirname(MODELS_PATH), exist_ok=True)
    joblib.dump(artifact, MODELS_PATH)
    return artifact


# ── Load artifact ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models… (first run trains on synthetic data, ~2 min)")
def load_artifact():
    if os.path.exists(MODELS_PATH):
        return joblib.load(MODELS_PATH)
    return _run_training()


def not_trained_banner():
    st.stop()


# ── Sidebar navigation ─────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛡️ Cognitive Anomaly Shield")
    st.caption("Explainable AI for Credit-Card Fraud Detection")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Summary", "Learn", "Models & Performance", "Try a Transaction", "SHAP Explanations"],
        index=0,
    )
    st.divider()
    st.caption("K. Jaswanth Kumar · K.V. Rohit Reddy · G. Hemanth")


# ── Helper colours ─────────────────────────────────────────────────────────

RED   = "#EF4444"
GREEN = "#22C55E"
BLUE  = "#3B82F6"
AMBER = "#F59E0B"


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Summary
# ═══════════════════════════════════════════════════════════════════════════

if page == "Summary":
    art = load_artifact()
    if art is None:
        not_trained_banner()

    st.title("Summary")
    st.markdown("Results on the **unseen test set**.")

    lr  = art["lr_results"]
    xgb = art["xgb_results"]
    n   = art["n_test"]
    nf  = art["n_frauds"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Test transactions", f"{n:,}")
    col2.metric("Frauds in test set", f"{nf:,}")
    col3.metric("Fraud rate", f"{art['fraud_rate']*100:.2f}%")

    st.divider()

    st.subheader("Model comparison")
    comparison = pd.DataFrame({
        "Metric":    ["Frauds caught (TP)", "False alarms (FP)", "Missed frauds (FN)",
                      "Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"],
        "Logistic Regression": [
            lr["tp"], lr["fp"], lr["fn"],
            lr["precision"], lr["recall"], lr["f1"], lr["pr_auc"], lr["roc_auc"],
        ],
        "XGBoost": [
            xgb["tp"], xgb["fp"], xgb["fn"],
            xgb["precision"], xgb["recall"], xgb["f1"], xgb["pr_auc"], xgb["roc_auc"],
        ],
    })
    st.dataframe(comparison.set_index("Metric"), use_container_width=True)

    st.divider()
    st.subheader("PR-AUC — why it matters")
    st.info(
        f"A random guesser has PR-AUC ≈ {art['fraud_rate']:.4f} (the fraud rate). "
        f"XGBoost achieves **{xgb['pr_auc']:.3f}** — far above chance."
    )

    # Bar chart
    metrics = ["precision", "recall", "f1", "pr_auc"]
    labels  = ["Precision", "Recall", "F1", "PR-AUC"]
    fig = go.Figure(data=[
        go.Bar(name="Logistic Regression", x=labels, y=[lr[m]  for m in metrics],
               marker_color=BLUE, text=[f"{lr[m]:.3f}"  for m in metrics], textposition="outside"),
        go.Bar(name="XGBoost",             x=labels, y=[xgb[m] for m in metrics],
               marker_color=GREEN, text=[f"{xgb[m]:.3f}" for m in metrics], textposition="outside"),
    ])
    fig.update_layout(barmode="group", title="Precision / Recall / F1 / PR-AUC comparison",
                      yaxis_range=[0, 1.1], height=400,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Learn
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Learn":
    art = load_artifact()
    if art is None:
        not_trained_banner()

    st.title("Why accuracy alone is a trap")

    fraud_rate = art["fraud_rate"]
    naive_acc  = 1 - fraud_rate

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud rate in test set", f"{fraud_rate*100:.2f}%")
    c2.metric("Accuracy of 'always-legitimate' model", f"{naive_acc*100:.2f}%")
    c3.metric("Frauds caught by that model", "0 of {:,}".format(art["n_frauds"]))

    st.error(
        "A model that **never flags anything** would be {:.2f}% accurate yet catch **zero frauds**. "
        "Accuracy hides everything we care about.".format(naive_acc * 100)
    )

    st.markdown("""
### The metrics that matter

| Metric | What it measures |
|--------|-----------------|
| **Precision** | Of every transaction we flag as fraud, what fraction really is fraud? |
| **Recall** | Of every real fraud, what fraction did we catch? |
| **F1** | Harmonic mean of precision and recall — balances the two. |
| **PR-AUC** | How well the model *ranks* fraud above legitimate transactions (independent of threshold). |

### The trade-off
Lowering the decision threshold catches **more frauds** (higher recall) but also raises **more false alarms** (lower precision).
A bank must decide which cost it can tolerate.

### Why a time-ordered split?
Fraud patterns evolve. Training on January–June 2020 and testing on July–December 2020 mirrors
real deployment — the model never saw the future, so the results are honest.
""")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Models & Performance
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Models & Performance":
    art = load_artifact()
    if art is None:
        not_trained_banner()

    st.title("Models & Performance")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrices", "Threshold Slider", "Precision-Recall Curves"])

    # ── Tab 1: Confusion matrices ──────────────────────────────────────────
    with tab1:
        lr  = art["lr_results"]
        xgb = art["xgb_results"]

        def cm_fig(results, title, color):
            z = [[results["tn"], results["fp"]], [results["fn"], results["tp"]]]
            text = [
                [f"TN\n{results['tn']:,}", f"FP (false alarm)\n{results['fp']:,}"],
                [f"FN (missed fraud)\n{results['fn']:,}", f"TP (fraud caught)\n{results['tp']:,}"],
            ]
            fig = go.Figure(go.Heatmap(
                z=z, text=text, texttemplate="%{text}",
                colorscale=[[0, "#1e3a5f"], [1, color]],
                showscale=False,
                xgap=3, ygap=3,
            ))
            fig.update_layout(
                title=title,
                xaxis=dict(ticktext=["Predicted: Legit", "Predicted: Fraud"], tickvals=[0, 1]),
                yaxis=dict(ticktext=["Actual: Legit", "Actual: Fraud"], tickvals=[0, 1]),
                height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            return fig

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(cm_fig(lr,  "Logistic Regression", BLUE),  use_container_width=True)
            st.caption(f"Threshold: {lr['threshold']}")
        with c2:
            st.plotly_chart(cm_fig(xgb, "XGBoost", GREEN), use_container_width=True)
            st.caption(f"Threshold: {xgb['threshold']}")

    # ── Tab 2: Threshold slider ────────────────────────────────────────────
    with tab2:
        st.markdown("Move the slider to see how the decision threshold affects fraud caught vs. false alarms.")
        model_choice = st.radio("Model", ["Logistic Regression", "XGBoost"], horizontal=True)
        sweep = art["lr_sweep"] if model_choice == "Logistic Regression" else art["xgb_sweep"]
        tuned_t = art["lr_thresh"] if model_choice == "Logistic Regression" else art["xgb_thresh"]

        t_slider = st.slider("Decision threshold", 0.01, 0.99, float(tuned_t), 0.01)

        row = sweep.loc[np.abs(sweep["threshold"] - t_slider).idxmin()]

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Frauds caught", f"{int(row['frauds_caught']):,}")
        mc2.metric("False alarms",  f"{int(row['false_alarms']):,}")
        mc3.metric("Precision",     f"{row['precision']:.3f}")
        mc4.metric("Recall",        f"{row['recall']:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["frauds_caught"],
                                 name="Frauds caught", line=dict(color=GREEN, width=2)))
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["false_alarms"],
                                 name="False alarms", line=dict(color=RED, width=2), yaxis="y2"))
        fig.add_vline(x=t_slider, line_dash="dash", line_color=AMBER,
                      annotation_text=f"t={t_slider:.2f}", annotation_position="top left")
        fig.update_layout(
            title="Frauds caught vs. false alarms across thresholds",
            xaxis_title="Threshold",
            yaxis_title="Frauds caught",
            yaxis2=dict(title="False alarms", overlaying="y", side="right"),
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Precision-Recall curves ────────────────────────────────────
    with tab3:
        lr_sw  = art["lr_sweep"]
        xgb_sw = art["xgb_sweep"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lr_sw["recall"], y=lr_sw["precision"],
                                 mode="lines", name="Logistic Regression",
                                 line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=xgb_sw["recall"], y=xgb_sw["precision"],
                                 mode="lines", name="XGBoost",
                                 line=dict(color=GREEN, width=2)))
        fig.add_hline(y=art["fraud_rate"], line_dash="dot", line_color="gray",
                      annotation_text="Random baseline", annotation_position="bottom right")
        fig.update_layout(
            title="Precision-Recall Curves",
            xaxis_title="Recall", yaxis_title="Precision",
            height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Try a Transaction
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Try a Transaction":
    art = load_artifact()
    if art is None:
        not_trained_banner()

    st.title("Try a Transaction")
    st.markdown("Enter transaction details and see the model's prediction + explanation.")

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Transaction")
            amount = st.number_input("Amount ($)", min_value=0.01, max_value=5000.0,
                                     value=913.04, step=0.01)
            hour = st.slider("Hour of transaction (24h)", 0, 23, 22)
            is_weekend = st.checkbox("Weekend", value=False)
            category_display = st.selectbox(
                "Merchant category",
                options=list(CATEGORY_DISPLAY.keys()),
                format_func=lambda k: CATEGORY_DISPLAY[k],
                index=list(CATEGORY_DISPLAY.keys()).index("misc_pos"),
            )

        with col2:
            st.subheader("Cardholder")
            customer_age = st.slider("Age", 18, 85, 44)
            gender = st.radio("Gender", ["Female", "Male"], index=0, horizontal=True)
            city_population = st.number_input("Home city population",
                                              min_value=100, max_value=5_000_000,
                                              value=50_000, step=1000)
            distance_km = st.number_input("Distance to merchant (km)",
                                          min_value=0.0, max_value=2000.0,
                                          value=39.6, step=0.1)

        model_choice = st.selectbox("Model", ["XGBoost (recommended)", "Logistic Regression"])
        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

    if submitted:
        is_male = 1 if gender == "Male" else 0
        X_raw = build_single_transaction(
            amount=amount,
            hour=hour,
            is_weekend=int(is_weekend),
            category=category_display,
            customer_age=customer_age,
            is_male=is_male,
            city_population=float(city_population),
            distance_km=distance_km,
        )

        X_scaled = scale(X_raw, art["scaler"])

        use_xgb = "XGBoost" in model_choice
        model     = art["xgb"]       if use_xgb else art["lr"]
        explainer = art["xgb_explainer"] if use_xgb else art["lr_explainer"]
        threshold = art["xgb_thresh"] if use_xgb else art["lr_thresh"]
        model_name = "XGBoost" if use_xgb else "Logistic Regression"

        prob  = float(model.predict_proba(X_scaled)[0, 1])
        label = "FRAUDULENT" if prob >= threshold else "LEGITIMATE"

        # Display result
        st.divider()
        col_r, col_d = st.columns([1, 2])

        with col_r:
            if label == "FRAUDULENT":
                st.error(f"## 🚨 {label}")
            else:
                st.success(f"## ✅ {label}")
            st.metric("Fraud probability", f"{prob*100:.1f}%")
            st.metric("Threshold", f"{threshold:.3f}")
            st.metric("Model", model_name)

        with col_d:
            st.markdown("**Transaction summary**")
            is_night_flag = (hour >= 22) or (hour < 4)
            st.markdown(
                f"${amount:,.2f} at a **{CATEGORY_DISPLAY[category_display]}** merchant, "
                f"{'at night' if is_night_flag else 'during the day'} on a "
                f"{'weekend' if is_weekend else 'weekday'}; "
                f"cardholder is **{customer_age} years old**, "
                f"**{'male' if is_male else 'female'}**, "
                f"living **{distance_km:.1f} km** from the merchant."
            )

            # SHAP
            shap_vals = shap_values_single(explainer, X_scaled)
            reasons = plain_english_reasons(shap_vals, X_scaled, top_n=5)

            st.markdown("**Why the model decided this:**")
            for i, r in enumerate(reasons, 1):
                st.markdown(f"{i}. {r}")

        # Waterfall chart
        st.divider()
        st.subheader("Feature contribution waterfall")

        sv_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "label":   [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS],
            "shap":    shap_vals,
        }).sort_values("shap", key=abs, ascending=False).head(12)

        colors = [RED if v > 0 else BLUE for v in sv_df["shap"]]
        fig = go.Figure(go.Bar(
            x=sv_df["shap"],
            y=sv_df["label"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in sv_df["shap"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="SHAP values (positive = pushes toward FRAUD, negative = pushes away)",
            xaxis_title="SHAP value (log-odds)",
            height=500,
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — SHAP Explanations (global)
# ═══════════════════════════════════════════════════════════════════════════

elif page == "SHAP Explanations":
    art = load_artifact()
    if art is None:
        not_trained_banner()

    st.title("SHAP Explanations — Global Feature Importance")
    st.markdown(
        "Mean absolute SHAP value across the test sample. "
        "Features higher up matter more to the model."
    )

    model_choice = st.radio("Model", ["XGBoost", "Logistic Regression"], horizontal=True)
    imp = art["xgb_importance"] if model_choice == "XGBoost" else art["lr_importance"]

    top_n = st.slider("Show top N features", 5, 28, 15)
    imp_top = imp.head(top_n)

    fig = go.Figure(go.Bar(
        x=imp_top["importance"],
        y=imp_top["label"],
        orientation="h",
        marker_color=GREEN,
        text=[f"{v:.4f}" for v in imp_top["importance"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} features by mean |SHAP| — {model_choice}",
        xaxis_title="Mean |SHAP value| (log-odds)",
        height=max(400, top_n * 28),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("What SHAP measures")
    st.markdown("""
- **Positive SHAP value** → this feature pushed the model *toward* predicting fraud.
- **Negative SHAP value** → this feature pushed the model *away* from fraud.
- The contributions add up to the model's score for that transaction.
- SHAP explains the **model's decision**, not the ground truth. It cannot prove fraud.
""")

    st.subheader("Full importance table")
    st.dataframe(
        imp.rename(columns={"feature": "Feature key", "label": "Feature name",
                             "importance": "Mean |SHAP|"}),
        use_container_width=True,
        hide_index=True,
    )
