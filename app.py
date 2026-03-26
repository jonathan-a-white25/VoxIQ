"""
VoxIQ — Customer Intelligence Platform
=======================================
Pages:
  1. 🏠  Home        — landing / upload CSV, auto column detection, score
  2. 📊  Insights    — Customer Sentiment & Product Insights (Mockup 2)
  3. 🔬  Performance — Product Performance Analysis (Mockup 1)
"""

import os, sys, io, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="VoxIQ",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

css_path = os.path.join(BASE_DIR, "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Plotly theme ────────────────────────────────────────────────────────────
PL = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.8)",
    font_family="DM Mono",
    font_color="#E2E8F0",
    margin=dict(l=20, r=20, t=45, b=20),
)
ACCENT  = "#00F5C4"
RED     = "#FF4D6D"
YELLOW  = "#FFD166"
BLUE    = "#00B4D8"
PURPLE  = "#7C6AF7"
GREEN   = "#4CAF82"
ORANGE  = "#FF8C42"

SENT_COLORS = {
    "positive": GREEN,
    "negative": RED,
    "neutral":  YELLOW,
    "unknown":  "#64748B",
}

# ── Column auto-detection ───────────────────────────────────────────────────
TEXT_HINTS    = ["text","review","body","content","comment","description",
                 "review_text","reviewtext","clean_text","feedback","message"]
RATING_HINTS  = ["rating","stars","score","star_rating","ratings","review_score"]
PRODUCT_HINTS = ["product","asin","parent_asin","product_id","item","sku",
                 "product_name","title","item_id","title_meta"]
DATE_HINTS    = ["date","review_date","created_at","timestamp","time","posted"]
TITLE_HINTS   = ["title_review","review_title","subject","headline"]

def detect_col(df, hints):
    cols_lower = {c.lower().replace(" ","_"): c for c in df.columns}
    for h in hints:
        if h in cols_lower:
            return cols_lower[h]
    return None

def auto_detect_columns(df):
    return {
        "text":    detect_col(df, TEXT_HINTS),
        "rating":  detect_col(df, RATING_HINTS),
        "product": detect_col(df, PRODUCT_HINTS),
        "date":    detect_col(df, DATE_HINTS),
        "title":   detect_col(df, TITLE_HINTS),
    }

# ── Scoring ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_from_s3():
    """Download fine-tuned DistilBERT from S3 and load it."""
    import boto3, tempfile
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    import torch

    bucket  = "operationcapstone-models"
    prefix  = "distilbert_finetuned/"
    files   = ["config.json","model.safetensors","tokenizer_config.json",
               "special_tokens_map.json","vocab.txt","tokenizer.json"]

    tmp_dir = tempfile.mkdtemp()
    s3      = boto3.client("s3")
    for f in files:
        s3.download_file(bucket, prefix + f, os.path.join(tmp_dir, f))

    tokenizer = DistilBertTokenizerFast.from_pretrained(tmp_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(tmp_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    return tokenizer, model, device

def score_dataframe(df, text_col, batch_size=128):
    """Score all rows in df[text_col] with the fine-tuned model."""
    import torch
    tokenizer, model, device = load_model_from_s3()
    texts = df[text_col].fillna("").astype(str).tolist()
    all_labels, all_conf, all_pos = [], [], []

    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i+batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True,
                           max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        labels     = np.argmax(probs, axis=1)
        confidence = probs[np.arange(len(labels)), labels]
        all_labels.extend(labels)
        all_conf.extend(confidence)
        all_pos.extend(probs[:, 1])

    df = df.copy()
    df["predicted_sentiment"] = ["positive" if l == 1 else "negative" for l in all_labels]
    df["confidence_score"]    = np.round(all_conf, 4)
    df["positive_probability"]= np.round(all_pos,  4)
    return df

@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

# ── Helpers ──────────────────────────────────────────────────────────────────
def fmt_k(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)

def section(label):
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)

def glow():
    st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="app-title" style="font-size:1.4rem">Vox<span>IQ</span></div>'
        '<div class="app-subtitle">Customer Intelligence Platform</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio("NAVIGATE", ["🏠  Home", "📊  Insights", "🔬  Performance"],
                    label_visibility="collapsed")

    st.markdown("---")

    # Active dataset info
    df_scored = st.session_state.get("df_scored")
    cols      = st.session_state.get("cols", {})
    if df_scored is not None:
        st.markdown('<div class="section-header">Active Dataset</div>', unsafe_allow_html=True)
        st.success(f"✓ {fmt_k(len(df_scored))} reviews scored")

        # Filters (only when dashboard pages are active)
        if page != "🏠  Home":
            st.markdown("---")
            st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

            # Sentiment filter
            st.markdown('<div style="font-size:0.65rem;color:#64748b;font-family:DM Mono,monospace;margin-bottom:0.3rem">SENTIMENT</div>', unsafe_allow_html=True)
            sent_opts = df_scored["predicted_sentiment"].dropna().unique().tolist()
            sel_sent  = []
            cols_sent  = st.columns(len(sent_opts))
            for i, s in enumerate(sent_opts):
                with cols_sent[i]:
                    if st.checkbox(s.capitalize(), value=True, key=f"sent_{s}"):
                        sel_sent.append(s)
            if not sel_sent:
                sel_sent = sent_opts

            # Rating filter
            if cols.get("rating") and cols["rating"] in df_scored.columns:
                st.markdown('<div style="font-size:0.65rem;color:#64748b;font-family:DM Mono,monospace;margin-top:0.8rem;margin-bottom:0.3rem">RATING</div>', unsafe_allow_html=True)
                all_ratings = sorted(df_scored[cols["rating"]].dropna().unique().tolist())
                sel_ratings = []
                for r in all_ratings:
                    if st.checkbox(f"{'⭐' * int(r) if r == int(r) else r}", value=True, key=f"rat_{r}"):
                        sel_ratings.append(r)
                if not sel_ratings:
                    sel_ratings = all_ratings
            else:
                sel_ratings = None

            # Product filter
            if cols.get("product") and cols["product"] in df_scored.columns:
                st.markdown('<div style="font-size:0.65rem;color:#64748b;font-family:DM Mono,monospace;margin-top:0.8rem;margin-bottom:0.3rem">PRODUCT</div>', unsafe_allow_html=True)
                all_products = sorted(df_scored[cols["product"]].dropna().astype(str).unique().tolist())
                sel_products = st.multiselect("", all_products, default=all_products[:],
                                              label_visibility="collapsed", key="prod_filter")
                if not sel_products:
                    sel_products = all_products
            else:
                sel_products = None

            st.session_state["sel_sent"]     = sel_sent
            st.session_state["sel_ratings"]  = sel_ratings
            st.session_state["sel_products"] = sel_products

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.6rem;color:#1e293b;font-family:DM Mono,monospace">'
        'VoxIQ v1.0 · Powered by DistilBERT</div>',
        unsafe_allow_html=True,
    )

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    '<div class="app-title">Vox<span>IQ</span></div>'
    '<div class="app-subtitle">// Customer Intelligence Platform · Powered by Fine-Tuned DistilBERT</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME / UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    # Hero
    st.markdown(
        '<div style="padding:2rem 0 1rem">'
        '<div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#E2E8F0">'
        'Turn customer reviews into <span style="color:#00F5C4">actionable intelligence.</span></div>'
        '<div style="font-family:DM Mono,monospace;font-size:0.85rem;color:#64748b;margin-top:0.5rem">'
        'Upload any review dataset. VoxIQ auto-detects your columns, scores every review with '
        'fine-tuned AI, and surfaces insights your team can act on immediately.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    glow()

    # Upload
    section("Upload Your Data")
    up = st.file_uploader("Upload a CSV file containing customer reviews",
                          type=["csv"], label_visibility="collapsed")

    if up:
        with st.spinner("Reading file…"):
            raw_df = load_csv(up.read())

        st.success(f"✓ {fmt_k(len(raw_df))} rows detected · {len(raw_df.columns)} columns")

        # Auto detect
        detected = auto_detect_columns(raw_df)

        section("Column Detection")
        st.markdown(
            '<div style="font-family:DM Mono,monospace;font-size:0.75rem;color:#64748b;margin-bottom:1rem">'
            'VoxIQ has automatically identified your columns. Confirm or adjust below.</div>',
            unsafe_allow_html=True,
        )

        all_cols     = [None] + list(raw_df.columns)
        all_cols_str = ["— not available —"] + list(raw_df.columns)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown('<div style="font-size:0.65rem;color:#00F5C4;font-family:DM Mono,monospace">REVIEW TEXT ✱</div>', unsafe_allow_html=True)
            text_col = st.selectbox("", raw_df.columns.tolist(),
                index=raw_df.columns.tolist().index(detected["text"]) if detected["text"] else 0,
                key="col_text", label_visibility="collapsed")
        with c2:
            st.markdown('<div style="font-size:0.65rem;color:#64748b;font-family:DM Mono,monospace">RATING</div>', unsafe_allow_html=True)
            rating_col = st.selectbox("", all_cols_str,
                index=all_cols.index(detected["rating"]) if detected["rating"] else 0,
                key="col_rating", label_visibility="collapsed")
            rating_col = None if rating_col == "— not available —" else rating_col
        with c3:
            st.markdown('<div style="font-size:0.65rem;color:#64748b;font-family:DM Mono,monospace">PRODUCT</div>', unsafe_allow_html=True)
            product_col = st.selectbox("", all_cols_str,
                index=all_cols.index(detected["product"]) if detected["product"] else 0,
                key="col_product", label_visibility="collapsed")
            product_col = None if product_col == "— not available —" else product_col
        with c4:
            st.markdown('<div style="font-size:0.65rem;color:#64748b;font-family:DM Mono,monospace">DATE</div>', unsafe_allow_html=True)
            date_col = st.selectbox("", all_cols_str,
                index=all_cols.index(detected["date"]) if detected["date"] else 0,
                key="col_date", label_visibility="collapsed")
            date_col = None if date_col == "— not available —" else date_col
        with c5:
            st.markdown('<div style="font-size:0.65rem;color:#64748b;font-family:DM Mono,monospace">REVIEW TITLE</div>', unsafe_allow_html=True)
            title_col = st.selectbox("", all_cols_str,
                index=all_cols.index(detected["title"]) if detected["title"] else 0,
                key="col_title", label_visibility="collapsed")
            title_col = None if title_col == "— not available —" else title_col

        st.markdown("---")

        # Preview
        with st.expander("📋 Preview uploaded data (first 5 rows)"):
            st.dataframe(raw_df.head(), use_container_width=True)

        # Score button
        score_btn = st.button("🚀  SCORE WITH VoxIQ AI", use_container_width=True)

        if score_btn:
            if not text_col:
                st.error("Please select a Review Text column before scoring.")
            else:
                prog = st.progress(0, text="Loading VoxIQ AI model from S3…")
                try:
                    prog.progress(20, text="Model loaded. Scoring reviews…")
                    scored = score_dataframe(raw_df, text_col)
                    prog.progress(90, text="Finalizing…")

                    # Store cols mapping
                    cols_map = {
                        "text":    text_col,
                        "rating":  rating_col,
                        "product": product_col,
                        "date":    date_col,
                        "title":   title_col,
                    }
                    st.session_state["df_scored"] = scored
                    st.session_state["cols"]      = cols_map
                    prog.progress(100, text="Done!")

                    pos = (scored["predicted_sentiment"] == "positive").sum()
                    neg = (scored["predicted_sentiment"] == "negative").sum()
                    avg_conf = scored["confidence_score"].mean()

                    st.markdown("---")
                    section("Scoring Complete")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Scored",      fmt_k(len(scored)))
                    m2.metric("Positive",          fmt_k(pos))
                    m3.metric("Negative",          fmt_k(neg))
                    m4.metric("Avg Confidence",    f"{avg_conf*100:.1f}%")

                    st.success("✓ Scoring complete. Navigate to **📊 Insights** or **🔬 Performance** in the sidebar.")

                    # Download
                    csv_out = scored.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️  Download Scored CSV",
                        csv_out,
                        file_name="voxiq_scored.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Scoring failed: {e}")
                    st.warning("Make sure AWS credentials are configured and the model is available in S3.")

    else:
        # Empty state
        st.markdown(
            '<div style="text-align:center;padding:4rem 2rem;color:#334155;'
            'font-family:DM Mono,monospace;font-size:0.85rem">'
            '⬆️  Upload a CSV file above to get started<span class="blink">_</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Feature cards
        section("What VoxIQ Does")
        f1, f2, f3 = st.columns(3)
        features = [
            ("🤖 AI Scoring", ACCENT, "Fine-tuned DistilBERT scores every review with 95%+ accuracy. No setup, no training required."),
            ("📊 Live Dashboard", YELLOW, "Interactive insights that update instantly. Filter by sentiment, rating, or product."),
            ("⚡ Auto Pipeline", PURPLE, "Drop files into your S3 folder and the dashboard updates automatically. No manual uploads needed."),
        ]
        for col, (title, color, desc) in zip([f1, f2, f3], features):
            with col:
                st.markdown(
                    f'<div class="card" style="border-top:3px solid {color};text-align:center;min-height:160px">'
                    f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:{color};margin-bottom:0.8rem">{title}</div>'
                    f'<div style="font-family:DM Mono,monospace;font-size:0.75rem;color:#94a3b8;line-height:1.6">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — apply filters to scored df
# ══════════════════════════════════════════════════════════════════════════════
def apply_filters(df, cols):
    sel_sent     = st.session_state.get("sel_sent",     df["predicted_sentiment"].unique().tolist())
    sel_ratings  = st.session_state.get("sel_ratings",  None)
    sel_products = st.session_state.get("sel_products", None)

    mask = df["predicted_sentiment"].isin(sel_sent)

    if sel_ratings is not None and cols.get("rating") and cols["rating"] in df.columns:
        mask &= df[cols["rating"]].isin(sel_ratings)

    if sel_products is not None and cols.get("product") and cols["product"] in df.columns:
        mask &= df[cols["product"]].astype(str).isin(sel_products)

    return df[mask].copy()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INSIGHTS (Mockup 2)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Insights":
    if st.session_state.get("df_scored") is None:
        st.info("⬅️  Upload and score a dataset on the Home page first.")
        st.stop()

    df   = st.session_state["df_scored"]
    cols = st.session_state.get("cols", {})
    fdf  = apply_filters(df, cols)

    section("Customer Sentiment & Product Insights")

    # KPI row
    total    = len(fdf)
    avg_rat  = fdf[cols["rating"]].mean() if cols.get("rating") and cols["rating"] in fdf.columns else None
    avg_conf = fdf["confidence_score"].mean()
    pos_pct  = (fdf["predicted_sentiment"] == "positive").sum() / total * 100 if total else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Reviews",      fmt_k(total))
    k2.metric("Average Rating",     f"{avg_rat:.2f}" if avg_rat is not None else "—")
    k3.metric("Avg Sentiment Score", f"{avg_conf:.2f}")
    k4.metric("Positive",           f"{pos_pct:.1f}%")

    glow()

    # Row 1 — Ratings Distribution + Sentiment Breakdown
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        if cols.get("rating") and cols["rating"] in fdf.columns:
            rc = fdf[cols["rating"]].value_counts().sort_index()
            bar_colors = []
            for r in rc.index:
                if   r <= 1: bar_colors.append(RED)
                elif r <= 2: bar_colors.append(ORANGE)
                elif r <= 3: bar_colors.append(YELLOW)
                elif r <= 4: bar_colors.append(GREEN)
                else:        bar_colors.append(ACCENT)
            fig = go.Figure(go.Bar(
                x=rc.index.astype(str), y=rc.values,
                marker_color=bar_colors,
                text=[fmt_k(v) for v in rc.values],
                textposition="outside",
            ))
            fig.update_layout(title="Ratings Distribution",
                              xaxis_title="Rating", yaxis_title="Count", **PL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rating column detected.")

    with r1c2:
        sc   = fdf["predicted_sentiment"].value_counts()
        fig  = go.Figure(go.Pie(
            labels=sc.index, values=sc.values, hole=0.6,
            marker_colors=[SENT_COLORS.get(s, "#64748B") for s in sc.index],
            textinfo="label+percent", textfont_size=11,
        ))
        pos_val = (fdf["predicted_sentiment"] == "positive").sum()
        fig.add_annotation(
            text=f"<b>{fmt_k(pos_val)}</b><br><span style='font-size:10px'>Positive</span>",
            x=0.5, y=0.5, showarrow=False, font_size=14, font_color=ACCENT
        )
        fig.update_layout(title="Sentiment Breakdown", **PL,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 — Top Products by Volume + Sentiment by Rating
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        if cols.get("product") and cols["product"] in fdf.columns:
            top_prod = (fdf.groupby(cols["product"])
                          .size().reset_index(name="count")
                          .sort_values("count", ascending=False).head(10))
            fig = go.Figure(go.Bar(
                y=top_prod[cols["product"]].astype(str),
                x=top_prod["count"],
                orientation="h",
                marker_color=BLUE,
                text=top_prod["count"].apply(fmt_k),
                textposition="outside",
            ))
            fig.update_layout(title="Top Products by Review Volume",
                              xaxis_title="Review Count",
                              yaxis=dict(autorange="reversed"), **PL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product column detected.")

    with r2c2:
        if cols.get("rating") and cols["rating"] in fdf.columns:
            grp = (fdf.groupby([cols["rating"], "predicted_sentiment"])
                      .size().reset_index(name="count"))
            fig = go.Figure()
            for sent, color in SENT_COLORS.items():
                d = grp[grp["predicted_sentiment"] == sent]
                if not d.empty:
                    fig.add_trace(go.Bar(
                        name=sent.capitalize(),
                        x=d[cols["rating"]].astype(str),
                        y=d["count"],
                        marker_color=color,
                        text=d["count"].apply(fmt_k),
                        textposition="outside",
                    ))
            fig.update_layout(barmode="stack", title="Sentiment by Rating",
                              xaxis_title="Rating", yaxis_title="Count", **PL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rating column detected.")

    # Row 3 — Confidence Distribution + Flagged Negatives
    glow()
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        fig = px.histogram(
            fdf, x="confidence_score", nbins=50, color="predicted_sentiment",
            color_discrete_map=SENT_COLORS,
            title="Confidence Score Distribution",
            labels={"confidence_score": "Confidence Score"},
            opacity=0.75,
        )
        fig.update_layout(barmode="overlay", **PL)
        st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        section("Flagged Negative Reviews")
        flagged = (fdf[fdf["predicted_sentiment"] == "negative"]
                     .sort_values("confidence_score", ascending=False)
                     .head(8))
        if not flagged.empty:
            for _, row in flagged.iterrows():
                text_preview = str(row[cols["text"]])[:120] + "…" if len(str(row[cols["text"]])) > 120 else str(row[cols["text"]])
                prod_label   = str(row[cols["product"]]) if cols.get("product") and cols["product"] in row else ""
                st.markdown(
                    f'<div class="card card-negative" style="padding:0.8rem;margin-bottom:0.5rem">'
                    f'<div style="font-size:0.6rem;color:#64748b;font-family:DM Mono,monospace">'
                    f'{prod_label} · {row["confidence_score"]*100:.0f}% confidence</div>'
                    f'<div style="font-size:0.75rem;color:#E2E8F0;font-family:DM Mono,monospace;margin-top:0.3rem">'
                    f'{text_preview}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No negative reviews in current filter.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE (Mockup 1)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Performance":
    if st.session_state.get("df_scored") is None:
        st.info("⬅️  Upload and score a dataset on the Home page first.")
        st.stop()

    df   = st.session_state["df_scored"]
    cols = st.session_state.get("cols", {})
    fdf  = apply_filters(df, cols)

    section("Product Performance Analysis & Review Insights")
    glow()

    prod_col   = cols.get("product")
    rating_col = cols.get("rating")

    if not prod_col or prod_col not in fdf.columns:
        st.warning("No product column detected in your dataset. This page requires product data.")
        st.stop()

    # Row 1 — Avg Rating by Product + Sentiment by Product
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        if rating_col and rating_col in fdf.columns:
            avg_rat = (fdf.groupby(prod_col)[rating_col]
                         .mean().reset_index()
                         .sort_values(rating_col, ascending=False).head(10))
            fig = go.Figure(go.Bar(
                y=avg_rat[prod_col].astype(str),
                x=avg_rat[rating_col].round(2),
                orientation="h",
                marker_color=GREEN,
                text=avg_rat[rating_col].round(2),
                textposition="outside",
            ))
            fig.update_layout(title="Average Rating by Product",
                              xaxis_title="Rating",
                              yaxis=dict(autorange="reversed"), **PL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rating column detected.")

    with r1c2:
        sent_prod = (fdf.groupby([prod_col, "predicted_sentiment"])
                       .size().reset_index(name="count"))
        top10 = (fdf.groupby(prod_col).size()
                   .sort_values(ascending=False).head(10).index.tolist())
        sent_prod = sent_prod[sent_prod[prod_col].isin(top10)]
        fig = go.Figure()
        for sent, color in SENT_COLORS.items():
            d = sent_prod[sent_prod["predicted_sentiment"] == sent]
            if not d.empty:
                fig.add_trace(go.Bar(
                    name=sent.capitalize(),
                    y=d[prod_col].astype(str),
                    x=d["count"],
                    orientation="h",
                    marker_color=color,
                    text=d["count"].apply(fmt_k),
                    textposition="outside",
                ))
        fig.update_layout(barmode="stack", title="Sentiment by Product",
                          xaxis_title="Review Count",
                          yaxis=dict(autorange="reversed"), **PL)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 — Confidence vs Rating scatter + Top Positive Products
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        if rating_col and rating_col in fdf.columns:
            sample = fdf.sample(min(3000, len(fdf)), random_state=42)
            fig = px.scatter(
                sample, x=rating_col, y="confidence_score",
                color="predicted_sentiment",
                color_discrete_map=SENT_COLORS,
                opacity=0.55,
                trendline="ols",
                title="Confidence Score vs Rating",
                labels={rating_col: "Star Rating", "confidence_score": "Confidence Score"},
            )
            fig.update_traces(marker_size=4)
            fig.update_layout(**PL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rating column detected.")

    with r2c2:
        pos_prod = (fdf[fdf["predicted_sentiment"] == "positive"]
                      .groupby(prod_col).size().reset_index(name="positive_count")
                      .sort_values("positive_count", ascending=False).head(10))
        fig = go.Figure(go.Bar(
            y=pos_prod[prod_col].astype(str),
            x=pos_prod["positive_count"],
            orientation="h",
            marker_color=ACCENT,
            text=pos_prod["positive_count"].apply(fmt_k),
            textposition="outside",
        ))
        fig.update_layout(title="Top Positive Products",
                          xaxis_title="Positive Reviews",
                          yaxis=dict(autorange="reversed"), **PL)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3 — Product table
    glow()
    section("Product Association Table")

    prod_summary = fdf.groupby(prod_col).agg(
        review_count=(prod_col, "count"),
        avg_rating=(rating_col, "mean") if rating_col and rating_col in fdf.columns else (prod_col, "count"),
        pct_positive=("predicted_sentiment", lambda x: round((x == "positive").sum() / len(x) * 100, 1)),
        avg_confidence=("confidence_score", "mean"),
    ).reset_index().sort_values("review_count", ascending=False).head(50)

    prod_summary.columns = [
        "Product", "Reviews", "Avg Rating" if rating_col else "Reviews",
        "% Positive", "Avg Confidence"
    ]
    prod_summary["Avg Confidence"] = (prod_summary["Avg Confidence"] * 100).round(1).astype(str) + "%"
    prod_summary["% Positive"]     = prod_summary["% Positive"].astype(str) + "%"
    if rating_col:
        prod_summary["Avg Rating"]  = prod_summary["Avg Rating"].round(2)

    st.dataframe(
        prod_summary,
        use_container_width=True,
        hide_index=True,
    )

    # Download
    st.markdown("---")
    csv_out = fdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️  Download Filtered Dataset",
        csv_out,
        file_name="voxiq_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
