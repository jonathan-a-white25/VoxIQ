"""
VoxIQ — Customer Intelligence Platform v4
==========================================
Pages:
  1. Overview    — upload CSV, auto column detection, score (session persistent)
  2. Insights    — Customer Sentiment & Product Insights
  3. Performance — Product Performance Analysis
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# These are the external libraries the app depends on.
#   - os, io, re, json: standard Python tools for file paths, bytes, regex, JSON
#   - numpy / pandas: numerical computing and dataframe manipulation
#   - streamlit: the web framework that renders every UI element
#   - plotly: interactive charts (bar, pie, heatmap, etc.)
# ══════════════════════════════════════════════════════════════════════════════
import os, io, re, json
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── App configuration ─────────────────────────────────────────────────────────
# This must be the FIRST Streamlit call in the file or it will throw an error.
# It sets the browser tab title, icon, and makes the app use the full screen width.
st.set_page_config(
    page_title="VoxIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS / Styling ─────────────────────────────────────────────────────────────
# Loads our custom stylesheet (style.css) and injects it into the page.
# This is how we override Streamlit's default look — fonts, colors, card styles, etc.
# unsafe_allow_html=True is required any time we inject raw HTML or CSS.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(BASE_DIR, "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
# PL is a shared dictionary of chart styling options passed to every Plotly chart.
# Instead of repeating these settings on every chart, we define them once and
# unpack them with **PL. This keeps all charts visually consistent.
PL = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",    # transparent background so CSS shines through
    plot_bgcolor="rgba(248,250,252,0.8)",
    font_family="Inter, sans-serif",
    font_color="#1E293B",
    margin=dict(l=20, r=20, t=45, b=20),
)

# ── Color constants ───────────────────────────────────────────────────────────
# Named hex colors used throughout the UI and charts.
# Centralizing these means we only change one line if we want to update a color.
ACCENT = "#3B82F6"  # blue — primary brand color
GREEN  = "#10B981"  # positive sentiment
RED    = "#EF4444"  # negative sentiment
YELLOW = "#F59E0B"  # neutral / warning
PURPLE = "#8B5CF6"
TEAL   = "#06B6D4"
ORANGE = "#F97316"

# Maps sentiment labels to their display colors in charts and cards.
SENT_COLORS = {
    "positive": GREEN,
    "negative": RED,
    "neutral":  YELLOW,
    "unknown":  "#94A3B8",
}

# ── Stopwords ─────────────────────────────────────────────────────────────────
# Words to exclude from the keyword frequency chart on the Insights page.
# Without this, the most common words would be "the", "a", "it", etc. —
# meaningless for sentiment analysis. We also include platform-specific noise
# like "amazon", "yelp", "check-in" so those don't dominate the results.
STOPWORDS = {
    "i","me","my","we","our","you","your","it","its","the","a","an","and","or",
    "but","in","on","at","to","for","of","with","this","that","is","was","are",
    "were","be","been","have","has","had","do","did","not","no","so","as","if",
    "by","from","up","out","about","than","then","they","them","their","what",
    "which","who","will","would","could","should","there","here","he","she","his",
    "her","just","also","very","more","how","all","its","been","into","through",
    "some","can","one","when","where","get","got","go","s","t","re","ve","ll",
    "product","item","order","bought","buy","amazon","review","stars","star","check","checkin","listed","yelp","place","visit","visited"
}

# ── Column auto-detection ─────────────────────────────────────────────────────
# When a user uploads a CSV, we don't know what their columns are named.
# These hint lists are our "vocabulary" for guessing which column serves which role.
# For example, if someone has a column called "review_body", we match it to TEXT_HINTS.
# The lists cover common naming conventions from Amazon, Yelp, Google, and custom exports.
TEXT_HINTS    = ["text","review","body","content","comment","description",
                 "review_text","reviewtext","clean_text","feedback","message",
                 "full_review_text","review_body","reviewbody","review_content",
                 "comments","comment_text","user_review","customer_review"]
RATING_HINTS  = ["rating","stars","score","star_rating","ratings","review_score",
                 "star_rating","num_stars","overall","overall_rating","score",
                 "rate","rated","review_rating","avg_rating","user_rating"]
PRODUCT_HINTS = ["title_meta","product_name","product","item_name",
                 "item","sku","asin","parent_asin","product_id","item_id",
                 "business_name","business","restaurant","hotel","location",
                 "coffee_shop_name","shop_name","store","store_name","brand",
                 "company","service","venue","place_name","listing",
                 "city","branch","outlet","region"]
DATE_HINTS    = ["date","review_date","created_at","timestamp","time","posted",
                 "date_posted","review_time","post_date","submission_date",
                 "created","updated","modified","published","review_created"]
TITLE_HINTS    = ["title_review","review_title","subject","headline","summary"]
REVIEWER_HINTS = ["name","reviewer","reviewer_name","username","user","author","customer"]

def detect_col(df, hints, exclude=None):
    """
    Tries to find which column in the dataframe matches a given role (text, rating, etc.)
    by comparing column names against the hint lists above.

    Step 1 — Exact match: normalizes column names to lowercase with underscores,
             then checks if any hint matches exactly.
    Step 2 — Fuzzy fallback: checks if any hint is a substring of a column name,
             or vice versa. Catches cases like "review_text_clean" matching "review_text".
    The `exclude` set prevents reviewer name columns from being accidentally matched
    to other roles (e.g., a "name" column being picked as the product column).
    """
    exclude = exclude or set()
    cols_lower = {c.lower().replace(" ","_"): c for c in df.columns if c not in exclude}
    for h in hints:
        if h in cols_lower:
            return cols_lower[h]
    # Fuzzy fallback — check if any hint is a substring of a column name
    for h in hints:
        for col_norm, col_orig in cols_lower.items():
            if h in col_norm or col_norm in h:
                return col_orig
    return None

def auto_detect_columns(df):
    """
    Runs detect_col() for each role and returns a dict of best guesses.
    Also includes two content-based fallbacks for when hint matching fails:
      - Text fallback: if no text column found by name, look for the column
        with the longest average string length (likely the review body).
      - Rating fallback: if no rating column found by name, look for a column
        whose values are all numbers between 1-5 with 10 or fewer unique values.

    Reviewer columns (name, author, username) are identified first and excluded
    from all other matching — otherwise "name" could match as a product column.
    """
    # Identify reviewer columns first so they are never matched to other roles
    reviewer_cols = set()
    cols_lower = {c.lower().replace(" ","_"): c for c in df.columns}
    for h in REVIEWER_HINTS:
        for col_norm, col_orig in cols_lower.items():
            if h == col_norm or h in col_norm:
                reviewer_cols.add(col_orig)

    detected = {
        "text":    detect_col(df, TEXT_HINTS,    exclude=reviewer_cols),
        "rating":  detect_col(df, RATING_HINTS,  exclude=reviewer_cols),
        "product": detect_col(df, PRODUCT_HINTS, exclude=reviewer_cols),
        "date":    detect_col(df, DATE_HINTS,    exclude=reviewer_cols),
        "title":   detect_col(df, TITLE_HINTS,   exclude=reviewer_cols),
    }
    # Content-based fallback: if no text column found by name, pick the longest column
    if not detected["text"]:
        for col in df.columns:
            if col in reviewer_cols:
                continue
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > 80:
                detected["text"] = col
                break
    # Content-based fallback: if no rating column found by name, look for a 1-5 numeric column
    if not detected["rating"]:
        for col in df.columns:
            if col in reviewer_cols:
                continue
            try:
                vals = pd.to_numeric(df[col].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
                if vals.dropna().between(1, 5).all() and vals.nunique() <= 10:
                    detected["rating"] = col
                    break
            except Exception:
                pass
    return detected

# ── UI helper functions ───────────────────────────────────────────────────────
# Small reusable utilities used throughout all three pages.

def truncate_label(val, max_len=30):
    """Cuts long product/location names down so they don't overflow chart labels."""
    s = str(val)
    return s[:max_len] + "..." if len(s) > max_len else s

def fmt_k(n):
    """Formats large numbers for display: 1500 → '1.5K', 2000000 → '2.0M'."""
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(int(n))

def section(label):
    """Renders a styled section header using a CSS class defined in style.css."""
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)

def divider():
    """Renders a horizontal visual divider between sections."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

def kpi(label, value, col, color=None):
    """
    Renders a single KPI metric card into a Streamlit column.
    Used to display summary stats like 'Total Reviews', 'Avg Rating', etc.
    The card is styled with a large colored value and a smaller gray label below it.
    """
    c = color or ACCENT
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value" style="color:{c}">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# This is the core of VoxIQ — the AI model that reads reviews and predicts sentiment.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model_from_s3():
    """
    Downloads our fine-tuned DistilBERT model from S3 and loads it into memory.

    @st.cache_resource means this function only runs ONCE per server session —
    the model stays loaded in RAM and is reused for every subsequent scoring run.
    Without this, we'd re-download the model on every page interaction (very slow).

    Flow:
      1. Connect to S3 and download the 6 model files (weights, config, tokenizer vocab)
         into a local cache directory on the server (~/.voxiq_model_cache/).
      2. Skip any file that already exists — so restarts are fast.
      3. Load the tokenizer (converts text to token IDs) and the model (the neural network).
      4. Set the model to eval() mode — disables dropout layers used during training.
      5. Move everything to GPU if available, otherwise CPU.
    """
    import boto3
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    import torch

    bucket   = "operationcapstone-models"
    prefix   = "distilbert_finetuned/"
    files    = ["config.json","model.safetensors","tokenizer_config.json",
                "special_tokens_map.json","vocab.txt","tokenizer.json"]
    # Use a permanent cache directory instead of /tmp
    cache_dir = os.path.expanduser("~/voxiq_model_cache")
    os.makedirs(cache_dir, exist_ok=True)

    s3 = boto3.client("s3")
    for f in files:
        dest = os.path.join(cache_dir, f)
        if not os.path.exists(dest):
            s3.download_file(bucket, prefix + f, dest)

    tokenizer = DistilBertTokenizerFast.from_pretrained(cache_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(cache_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model.to(device), device

def score_dataframe(df, text_col, batch_size=64):
    """
    Runs every review through the DistilBERT model and adds three new columns:
      - predicted_sentiment:  'positive' or 'negative'
      - confidence_score:     how certain the model is (0.0 – 1.0)
      - positive_probability: raw probability of being positive (0.0 – 1.0)

    Why batches?
      The model can't process 50,000 reviews at once — it would run out of memory.
      We send 64 reviews at a time (one batch), get predictions back, and accumulate
      results in lists. The progress bar updates after each batch.

    What torch.no_grad() does:
      During training, PyTorch tracks every calculation to compute gradients.
      We don't need gradients when making predictions — disabling them saves memory
      and makes inference significantly faster.

    What softmax does:
      The model outputs raw "logit" scores (e.g., [-1.2, 2.4]). Softmax converts
      these into proper probabilities that sum to 1.0 (e.g., [0.05, 0.95]).
      The higher value is our predicted class; its probability is the confidence score.
    """
    import torch
    tokenizer, model, device = load_model_from_s3()
    texts = df[text_col].fillna("").astype(str).tolist()
    all_labels, all_conf, all_pos = [], [], []

    progress = st.progress(0, text="Model loaded. Scoring reviews...")
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i+batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True,
                           max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs      = torch.softmax(logits, dim=1).cpu().numpy()
        labels     = np.argmax(probs, axis=1)
        confidence = probs[np.arange(len(labels)), labels]
        all_labels.extend(labels)
        all_conf.extend(confidence)
        all_pos.extend(probs[:, 1])

        batch_num = (i // batch_size) + 1
        progress.progress(int(batch_num / total_batches * 100),
                          text=f"Scoring... {min(i+batch_size, len(texts)):,} / {len(texts):,} reviews")

    progress.empty()
    df = df.copy()
    df["predicted_sentiment"]  = ["positive" if l == 1 else "negative" for l in all_labels]
    df["confidence_score"]     = np.round(all_conf, 4)
    df["positive_probability"] = np.round(all_pos,  4)
    return df

@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    """
    Reads the uploaded CSV file into a pandas DataFrame.
    @st.cache_data caches the result by file content — so if the same file is
    uploaded twice, it skips re-reading and returns the cached version instantly.
    We receive raw bytes from the uploader, wrap them in BytesIO, and pass to pandas.
    """
    return pd.read_csv(io.BytesIO(file_bytes))

def get_word_freq(texts, top_n=15):
    """
    Counts the most common meaningful words across a list of review texts.
    Used to power the 'Top Keywords by Sentiment' chart on the Insights page.

    Steps:
      1. For each review, extract all lowercase words that are 3+ characters long.
      2. Filter out stopwords (see STOPWORDS set above).
      3. Count everything using Counter and return the top N most frequent words.
    The regex r'\b[a-z]{3,}\b' means: word boundary, 3+ lowercase letters, word boundary.
    """
    words = []
    for t in texts:
        tokens = re.findall(r'\b[a-z]{3,}\b', str(t).lower())
        words.extend([w for w in tokens if w not in STOPWORDS])
    return Counter(words).most_common(top_n)

def clean_rating_col(df, rating_col):
    """
    Normalizes the rating column to plain numbers.
    Some datasets store ratings as strings like "4.0 star rating" or "3 stars".
    This function extracts just the numeric part using a regex, so the model
    and charts can treat ratings as actual numbers rather than text.
    """
    if rating_col and rating_col in df.columns:
        df = df.copy()
        extracted = df[rating_col].astype(str).str.extract(r'([\d.]+)')
        if extracted is not None and not extracted.empty:
            df[rating_col] = pd.to_numeric(extracted[0], errors='coerce')
    return df

def clean_text_prefixes(df, text_col):
    """
    Strips noise from the beginning of review text before it reaches the model.
    Some platforms (especially Yelp) embed metadata at the start of the text field:
      - Dates: "03/15/2023 Great coffee!" → strips the date, saves it separately
      - Check-in counts: "47 check-ins Great coffee!" → strips the number
      - Category tags: "Listed in Coffee, Breakfast Great coffee!" → strips the tag

    Why this matters: the model was trained on clean review text. Feeding it
    "47 check-ins" as the start of a review would confuse it and reduce accuracy.

    The date extraction is smart: if more than 5% of reviews have dates at the
    start, we save them in a separate '_extracted_date' column so the timeline
    chart can use them later. If fewer than 5% have dates, we just strip them.
    """
    if not text_col or text_col not in df.columns:
        return df, None
    df = df.copy()
    texts = df[text_col].astype(str)
    date_pattern = r"^(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})"
    extracted_dates = texts.str.extract(date_pattern)[0]
    has_dates = extracted_dates.notna().sum() > len(df) * 0.05  # 5% threshold
    texts = texts.str.replace(date_pattern, "", regex=True).str.strip()
    texts = texts.str.replace(r"^\d+\s+check-ins?\s*", "", regex=True).str.strip()
    texts = texts.str.replace(r"^Listed in[^.!?]*?,\s*", "", regex=True).str.strip()
    texts = texts.str.strip()
    df[text_col] = texts
    if has_dates:
        return df, extracted_dates
    return df, None

# ── Training data export ──────────────────────────────────────────────────────
def upload_training_data_to_s3(scored, text_col, filename):
    """
    After every scoring run, this function automatically feeds high-quality
    predictions back into S3 so the model can learn from them during retraining.

    This is the INPUT side of the automated retraining pipeline:
      VoxIQ scores data → this function uploads to S3/training_data/ →
      EventBridge triggers Lambda every Monday → SageMaker retrains on all
      accumulated CSVs → new model replaces the old one.

    Why only confidence >= 0.85?
      We only want high-confidence predictions for retraining. If the model was
      uncertain about a review (e.g., 60% positive), using it as training data
      could teach the model incorrect labels and degrade performance over time.
      At 85%+ confidence, we trust the prediction enough to treat it as ground truth.

    The upload is wrapped in try/except so a failed S3 upload never crashes the
    scoring flow — the user still gets their results, and we just show a warning.
    """
    try:
        import boto3
        from datetime import datetime, timezone
        high_conf = scored[scored["confidence_score"] >= 0.85][
            [text_col, "predicted_sentiment", "confidence_score"]
        ].rename(columns={text_col: "text"})
        if high_conf.empty:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        key = f"training_data/{timestamp}_{filename}"
        buf = io.BytesIO()
        high_conf.to_csv(buf, index=False)
        buf.seek(0)
        boto3.client("s3").put_object(
            Bucket="operationcapstone-models", Key=key, Body=buf.getvalue()
        )
    except Exception as e:
        st.warning(f"Training data upload skipped: {e}")

# ── Model version info ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_model_info():
    """
    Reads the model version log from S3 and returns the most recent entry.
    This powers the 'Model Info' section in the sidebar showing version number,
    retraining date, how many rows it was trained on, and its accuracy.

    The log is a JSON array where each entry is appended after a successful
    SageMaker retraining job. We grab the last item (most recent version).

    @st.cache_data(ttl=3600) means the result is cached for 1 hour.
    After an hour, the next user interaction fetches fresh data from S3.
    This avoids hitting S3 on every single page load while still staying
    reasonably up to date after a retraining job completes.
    """
    try:
        import boto3, json as _json
        obj = boto3.client("s3").get_object(
            Bucket="operationcapstone-models", Key="model_versions/log.json"
        )
        log = _json.loads(obj["Body"].read())
        return log[-1] if log else None
    except Exception:
        return None

# ── Mapping persistence ───────────────────────────────────────────────────────
# When a user corrects the auto-detected column mapping, we save it to a local
# JSON file keyed by the original filename. The next time they upload the same
# file, we load their saved mapping instead of re-running auto-detection.
# This means users never have to manually remap the same dataset twice.
MAPPINGS_PATH = os.path.join(BASE_DIR, "voxiq_mappings.json")

def load_saved_mapping(filename):
    if not os.path.exists(MAPPINGS_PATH):
        return None
    with open(MAPPINGS_PATH) as f:
        saved = json.load(f).get(filename)
    if saved is None:
        return None
    # Backward compat: migrate old "date"/"title" top-level keys to extras
    if "date" in saved or "title" in saved:
        extras = list(saved.get("extras", []))
        if saved.get("date"):
            extras.insert(0, {"label": "Date", "col": saved.pop("date")})
        if saved.get("title"):
            extras.append({"label": "Review Title", "col": saved.pop("title")})
        saved["extras"] = extras
    return saved

def save_mapping(filename, cols_map):
    mappings = {}
    if os.path.exists(MAPPINGS_PATH):
        with open(MAPPINGS_PATH) as f:
            mappings = json.load(f)
    entry = {}
    for k in ("text", "rating", "product"):
        if cols_map.get(k):
            entry[k] = cols_map[k]
    if cols_map.get("entity_label"):
        entry["entity_label"] = cols_map["entity_label"]
    if cols_map.get("extras"):
        entry["extras"] = [e for e in cols_map["extras"] if e.get("col")]
    mappings[filename] = entry
    with open(MAPPINGS_PATH, "w") as f:
        json.dump(mappings, f, indent=2)

# ── Filter helper ─────────────────────────────────────────────────────────────
def apply_filters(df, cols):
    """
    Reads the current filter selections from session state and returns a filtered
    copy of the scored dataframe. Every chart and metric on the Insights and
    Performance pages calls this first, so filters affect everything simultaneously.

    Filters applied in order:
      1. Sentiment    — show only positive, negative, or both
      2. Rating       — show only selected star ratings (e.g., just 1-star and 2-star)
      3. Product      — show only selected products/locations
      4. Keyword      — show only reviews containing a specific word or phrase
      5. Confidence   — hide reviews where the model was below a certainty threshold
      6. Score range  — narrow to reviews within a specific positive probability range

    We read filter values from st.session_state because the filter widgets (checkboxes,
    sliders, etc.) are defined in render_filters_panel() — a separate function that runs
    before this one. Streamlit stores widget values in session_state so they can be
    accessed anywhere in the script on the same page load.
    """
    fdf = df.copy()

    # Sentiment
    sel_sent = st.session_state.get("f_sent", fdf["predicted_sentiment"].unique().tolist())
    fdf = fdf[fdf["predicted_sentiment"].isin(sel_sent)]

    # Rating
    if cols.get("rating") and cols["rating"] in fdf.columns:
        sel_ratings = st.session_state.get("f_ratings", sorted(fdf[cols["rating"]].dropna().unique().tolist()))
        fdf = fdf[fdf[cols["rating"]].isin(sel_ratings)]

    # Product
    if cols.get("product") and cols["product"] in fdf.columns:
        sel_products = st.session_state.get("f_products", None)
        if sel_products:
            fdf = fdf[fdf[cols["product"]].astype(str).isin(sel_products)]

    # Keyword search
    kw = st.session_state.get("f_keyword", "").strip()
    if kw and cols.get("text") and cols["text"] in fdf.columns:
        fdf = fdf[fdf[cols["text"]].fillna("").str.contains(kw, case=False, na=False)]

    # Confidence threshold
    conf_min = st.session_state.get("f_conf_min", 0.0)
    fdf = fdf[fdf["confidence_score"] >= conf_min]

    # Sentiment score range
    prob_min = st.session_state.get("f_prob_min", 0.0)
    prob_max = st.session_state.get("f_prob_max", 1.0)
    fdf = fdf[(fdf["positive_probability"] >= prob_min) & (fdf["positive_probability"] <= prob_max)]

    return fdf.copy()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# The sidebar renders on every page. It contains:
#   - Navigation (Overview / Insights / Performance)
#   - Active dataset status + "Clear dataset" button
#   - Model version info pulled from S3
# The `page` variable set here controls which page block runs further down.
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">Vox<span>IQ</span></div>'
        '<div class="sidebar-tagline">Customer Intelligence Platform</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    page = st.radio("NAV", ["Overview", "Insights", "Performance"],
                    label_visibility="collapsed")
    entity_label = st.session_state.get("entity_label", "Product")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    df_scored = st.session_state.get("df_scored")
    if df_scored is not None:
        st.markdown('<div class="sidebar-section">Active Dataset</div>', unsafe_allow_html=True)
        st.success(f"{fmt_k(len(df_scored))} reviews scored")
        if st.button("Clear dataset", use_container_width=True, key="clear_btn"):
            for k in ["df_scored","cols","raw_df"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    model_info = fetch_model_info()
    if model_info:
        from datetime import datetime
        try:
            retrain_date = datetime.fromisoformat(model_info["timestamp"]).strftime("%b %d, %Y")
        except Exception:
            retrain_date = model_info.get("timestamp", "—")
        st.markdown(
            f'<div class="sidebar-section">Model Info</div>'
            f'<div class="sidebar-meta">v{model_info.get("version","—")} · Retrained {retrain_date}</div>'
            f'<div class="sidebar-meta">{model_info.get("rows_trained",0):,} rows · '
            f'{model_info.get("accuracy",0)*100:.1f}% acc</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sidebar-meta">Model info unavailable</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-footer">VoxIQ v1.0 · Powered by DistilBERT</div>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    '<div class="app-title">Vox<span>IQ</span></div>'
    '<div class="app-subtitle">Customer Intelligence Platform · Powered by Fine-Tuned DistilBERT</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Filters panel ─────────────────────────────────────────────────────────────
# This function renders the collapsible filter UI at the top of Insights and
# Performance pages. It writes filter selections into st.session_state so that
# apply_filters() (defined above) can read them and filter the dataframe.
# The two functions work as a pair: render_filters_panel() collects user input,
# apply_filters() acts on it. They're separated so the filter logic stays clean.
def render_filters_panel(df, cols):
    with st.expander("Filters", expanded=False):
        st.markdown('<div class="filter-label">Sentiment</div>', unsafe_allow_html=True)
        sent_opts = df["predicted_sentiment"].dropna().unique().tolist()
        sel_sent = []
        fcols = st.columns(len(sent_opts))
        for i, s in enumerate(sent_opts):
            with fcols[i]:
                if st.checkbox(s.capitalize(), value=True, key=f"f_sent_{s}"):
                    sel_sent.append(s)
        st.session_state["f_sent"] = sel_sent if sel_sent else sent_opts

        if cols.get("rating") and cols["rating"] in df.columns:
            st.markdown('<div class="filter-label">Star Rating</div>', unsafe_allow_html=True)
            all_ratings = sorted(df[cols["rating"]].dropna().unique().tolist())
            sel_ratings = []
            rcols = st.columns(len(all_ratings))
            for i, r in enumerate(all_ratings):
                with rcols[i]:
                    label = str(int(r)) if r == int(r) else str(r)
                    if st.checkbox(label, value=True, key=f"f_rat_{r}"):
                        sel_ratings.append(r)
            st.session_state["f_ratings"] = sel_ratings if sel_ratings else all_ratings

        if cols.get("product") and cols["product"] in df.columns:
            st.markdown('<div class="filter-label">Product</div>', unsafe_allow_html=True)
            all_products = sorted(df[cols["product"]].dropna().astype(str).unique().tolist())
            sel_products = st.multiselect("", all_products, default=[],
                                          placeholder="All products",
                                          label_visibility="collapsed", key="f_products_ms")
            st.session_state["f_products"] = sel_products if sel_products else None

        st.markdown('<div class="filter-label">Keyword Search</div>', unsafe_allow_html=True)
        kw = st.text_input("", placeholder="Search review text...",
                           label_visibility="collapsed", key="f_keyword_input")
        st.session_state["f_keyword"] = kw

        st.markdown('<div class="filter-label">Min Confidence</div>', unsafe_allow_html=True)
        conf_min = st.slider("", 0.0, 1.0, 0.0, 0.05,
                             label_visibility="collapsed", key="f_conf_slider")
        st.session_state["f_conf_min"] = conf_min

        st.markdown('<div class="filter-label">Sentiment Score Range</div>', unsafe_allow_html=True)
        prob_range = st.slider("", 0.0, 1.0, (0.0, 1.0), 0.05,
                               label_visibility="collapsed", key="f_prob_slider")
        st.session_state["f_prob_min"] = prob_range[0]
        st.session_state["f_prob_max"] = prob_range[1]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# This page handles the entire upload → map → score flow.
# It's "session persistent" meaning once a dataset is scored, navigating away
# and back does NOT clear it — the scored data stays in st.session_state["df_scored"]
# until the user explicitly clicks "Clear dataset" in the sidebar.
#
# Two states:
#   A) Dataset already scored → show summary and stop. Don't re-render the upload UI.
#   B) No dataset → show the upload flow from scratch.
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    df_scored = st.session_state.get("df_scored")
    cols      = st.session_state.get("cols", {})

    # STATE A: Dataset already in memory — show summary and skip the upload UI entirely
    if df_scored is not None:
        st.markdown(
            '<div class="hero-block">'
            '<div class="hero-title">Dataset loaded and scored.</div>'
            '<div class="hero-sub">Navigate to Insights or Performance to explore your data. '
            'To upload a new dataset, click "Clear dataset" in the sidebar.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        divider()
        section("Current Dataset Summary")
        pos      = (df_scored["predicted_sentiment"] == "positive").sum()
        neg      = (df_scored["predicted_sentiment"] == "negative").sum()
        avg_conf = df_scored["confidence_score"].mean()
        avg_rat  = df_scored[cols["rating"]].mean() if cols.get("rating") and cols["rating"] in df_scored.columns else None

        k1, k2, k3, k4, k5 = st.columns(5)
        kpi("Total Reviews",  fmt_k(len(df_scored)),                k1)
        kpi("Avg Rating",     f"{avg_rat:.2f}" if avg_rat else "—", k2)
        kpi("Positive",       fmt_k(pos),                           k3, GREEN)
        kpi("Negative",       fmt_k(neg),                           k4, RED)
        kpi("Avg Confidence", f"{avg_conf*100:.1f}%",               k5)

        csv_out = df_scored.to_csv(index=False).encode("utf-8")
        st.download_button("Download Scored CSV", csv_out,
                           file_name="voxiq_scored.csv", mime="text/csv",
                           use_container_width=True)
        st.stop()

    # STATE B: No dataset yet — show the upload and mapping UI
    st.markdown(
        '<div class="hero-block">'
        '<div class="hero-title">Turn customer reviews into <span>actionable intelligence.</span></div>'
        '<div class="hero-sub">Upload any review dataset. VoxIQ auto-detects your columns, scores every '
        'review with fine-tuned AI, and surfaces insights your team can act on immediately.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    divider()
    section("Upload Your Data")

    up = st.file_uploader("Upload a CSV file containing customer reviews",
                          type=["csv"], label_visibility="collapsed")

    if up:
        with st.spinner("Reading file..."):
            raw_df = load_csv(up.read())
        st.session_state["raw_df"] = raw_df
        st.success(f"{fmt_k(len(raw_df))} rows detected · {len(raw_df.columns)} columns")

        saved_mapping   = load_saved_mapping(up.name)
        mapping_loaded  = saved_mapping is not None
        detected        = saved_mapping if mapping_loaded else auto_detect_columns(raw_df)

        # Reset extra columns whenever a new file is uploaded
        if st.session_state.get("_last_mapped_file") != up.name:
            if mapping_loaded:
                st.session_state["extra_cols"] = saved_mapping.get("extras", [])
                if saved_mapping.get("entity_label"):
                    st.session_state["entity_label"] = saved_mapping["entity_label"]
            else:
                extras = []
                if detected.get("date"):
                    extras.append({"label": "Date", "col": detected["date"]})
                if detected.get("title"):
                    extras.append({"label": "Review Title", "col": detected["title"]})
                st.session_state["extra_cols"] = extras
            st.session_state["_last_mapped_file"] = up.name

        section("Column Mapping")
        if mapping_loaded:
            st.markdown(
                '<div class="helper-text">Mapping saved for this dataset</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="helper-text">VoxIQ has automatically identified your columns. '
                'Confirm or adjust below before scoring.</div>',
                unsafe_allow_html=True,
            )

        all_cols     = [None] + list(raw_df.columns)
        all_cols_str = ["— not available —"] + list(raw_df.columns)

        # ── Row 1: core columns ───────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            st.markdown('<div class="col-label">Review Text</div>', unsafe_allow_html=True)
            text_col = st.selectbox("", raw_df.columns.tolist(),
                index=raw_df.columns.tolist().index(detected["text"]) if detected.get("text") in raw_df.columns.tolist() else 0,
                key="col_text", label_visibility="collapsed")
        with c2:
            st.markdown('<div class="col-label">Rating</div>', unsafe_allow_html=True)
            rating_col = st.selectbox("", all_cols_str,
                index=all_cols.index(detected["rating"]) if detected.get("rating") in all_cols else 0,
                key="col_rating", label_visibility="collapsed")
            rating_col = None if rating_col == "— not available —" else rating_col
        with c3:
            entity_input = st.text_input("", value=st.session_state.get("entity_label", "Product"),
                placeholder="e.g. Product, Location, Hotel",
                label_visibility="collapsed", key="entity_label_input")
            st.session_state["entity_label"] = entity_input
            entity_col = st.selectbox("", all_cols_str,
                index=all_cols.index(detected["product"]) if detected.get("product") in all_cols else 0,
                key="col_product", label_visibility="collapsed")
            entity_col = None if entity_col == "— not available —" else entity_col
        with c4:
            st.markdown('<div class="col-label">&nbsp;</div>', unsafe_allow_html=True)
            if st.button("＋", use_container_width=True, help="Add another column"):
                if "extra_cols" not in st.session_state:
                    st.session_state["extra_cols"] = []
                st.session_state["extra_cols"].append({"label": "", "col": None})
                st.rerun()

        # ── Extra columns (below) ─────────────────────────────────────────────
        extra_cols_state = st.session_state.get("extra_cols", [])
        for i, extra in enumerate(extra_cols_state):
            ex1, ex2, ex3 = st.columns([1.5, 3.5, 0.6])
            with ex1:
                st.text_input("Label", value=extra.get("label", ""),
                              placeholder="Label",
                              key=f"extra_label_{i}", label_visibility="collapsed")
            with ex2:
                saved_col = extra.get("col")
                idx = all_cols.index(saved_col) if saved_col in all_cols else 0
                st.selectbox("Column", all_cols_str, index=idx,
                             key=f"extra_col_{i}", label_visibility="collapsed")
            with ex3:
                if st.button("✕", key=f"rm_extra_{i}", use_container_width=True):
                    st.session_state["extra_cols"].pop(i)
                    st.rerun()

        save_mapping_checked = st.checkbox("Save this mapping", value=mapping_loaded,
                                           key="save_mapping_cb")

        st.markdown("---")
        with st.expander("Preview uploaded data (first 5 rows)"):
            st.dataframe(raw_df.head(), use_container_width=True)

        if st.button("Score with VoxIQ AI", use_container_width=True):
            if not text_col:
                st.error("Please select a Review Text column before scoring.")
            else:
                try:
                    # ── Scoring flow ──────────────────────────────────────────
                    # 1. Collect extra column mappings from widget state
                    # 2. Clean the data (ratings, text prefixes)
                    # 3. Run DistilBERT inference
                    # 4. Upload high-confidence rows to S3 for future retraining
                    # 5. Store the scored dataframe and column map in session state
                    #    so Insights and Performance pages can access it immediately

                    # Build extras from widget state
                    n_extras = len(st.session_state.get("extra_cols", []))
                    extras_built = []
                    for i in range(n_extras):
                        lbl   = st.session_state.get(f"extra_label_{i}", "").strip()
                        col_v = st.session_state.get(f"extra_col_{i}", "— not available —")
                        col_v = None if col_v == "— not available —" else col_v
                        extras_built.append({"label": lbl or f"Extra {i+1}", "col": col_v})

                    raw_df = clean_rating_col(raw_df, rating_col)
                    raw_df, extracted_dates = clean_text_prefixes(raw_df, text_col)
                    # Use date from extras if present; otherwise fall back to extracted dates
                    date_col = next(
                        (e["col"] for e in extras_built if e.get("col") and "date" in e.get("label", "").lower()),
                        None
                    )
                    if extracted_dates is not None and not date_col:
                        date_col = "_extracted_date"
                        extras_built.append({"label": "Date", "col": "_extracted_date"})
                        st.session_state["extra_cols"].append({"label": "Date", "col": "_extracted_date"})
                    scored = score_dataframe(raw_df, text_col)
                    if extracted_dates is not None and date_col == "_extracted_date":
                        scored["_extracted_date"] = extracted_dates.values
                    upload_training_data_to_s3(scored, text_col, up.name)
                    cols_map = {
                        "text":         text_col,
                        "rating":       rating_col,
                        "product":      entity_col,
                        "entity_label": st.session_state.get("entity_label", "Product"),
                        "extras":       extras_built,
                    }
                    if save_mapping_checked:
                        save_mapping(up.name, cols_map)
                    st.session_state["df_scored"] = scored
                    st.session_state["cols"]      = cols_map

                    pos      = (scored["predicted_sentiment"] == "positive").sum()
                    neg      = (scored["predicted_sentiment"] == "negative").sum()
                    avg_conf = scored["confidence_score"].mean()

                    section("Scoring Complete")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Scored",   fmt_k(len(scored)))
                    m2.metric("Positive",       fmt_k(pos))
                    m3.metric("Negative",       fmt_k(neg))
                    m4.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
                    st.success("Scoring complete. Navigate to Insights or Performance in the sidebar.")

                    csv_out = scored.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Scored CSV", csv_out,
                                       file_name="voxiq_scored.csv", mime="text/csv",
                                       use_container_width=True)
                except Exception as e:
                    st.error(f"Scoring failed: {e}")
    else:
        st.markdown('<div class="empty-state">Upload a CSV file above to get started.</div>',
                    unsafe_allow_html=True)
        section("Platform Capabilities")
        f1, f2, f3 = st.columns(3)
        features = [
            ("AI-Powered Scoring", ACCENT,
             "Fine-tuned DistilBERT scores every review with 95%+ accuracy."),
            ("Interactive Dashboard", GREEN,
             "Live insights that update in real time. Filter by sentiment, rating, or product."),
            ("Automated Pipeline", PURPLE,
             "Connect your data source once. VoxIQ processes new reviews automatically."),
        ]
        for col, (title, color, desc) in zip([f1, f2, f3], features):
            with col:
                st.markdown(
                    f'<div class="feature-card" style="border-top:3px solid {color}">'
                    f'<div class="feature-title" style="color:{color}">{title}</div>'
                    f'<div class="feature-desc">{desc}</div>'
                    f'</div>', unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INSIGHTS
# Displays customer-level sentiment analysis. All charts here are filtered by
# the Filters panel at the top — the filtered dataframe `fdf` is what every
# chart and metric reads from.
#
# Charts on this page:
#   - KPI row: total, avg rating, positive %, negative %, confidence, net sentiment
#   - Ratings Distribution (bar chart)
#   - Sentiment Breakdown (donut chart)
#   - Top Entity by Review Volume (horizontal bar)
#   - Sentiment by Rating (stacked bar)
#   - Top Keywords by Sentiment (grouped horizontal bar)
#   - Top 5 Positive Reviews (cards)
#   - Top 6 Flagged Negative Reviews (cards, 2 columns)
# ══════════════════════════════════════════════════════════════════════════════
if page == "Insights":
    if st.session_state.get("df_scored") is None:
        st.info("Upload and score a dataset on the Overview page first.")
        st.stop()

    df   = st.session_state["df_scored"]
    cols = st.session_state.get("cols", {})
    entity_label = st.session_state.get("entity_label", "Product")

    render_filters_panel(df, cols)
    fdf = apply_filters(df, cols)

    section("Customer Sentiment & Product Insights")

    total    = len(fdf)
    avg_rat  = fdf[cols["rating"]].mean() if cols.get("rating") and cols["rating"] in fdf.columns else None
    avg_conf = fdf["confidence_score"].mean()
    pos_n    = (fdf["predicted_sentiment"] == "positive").sum()
    neg_n    = (fdf["predicted_sentiment"] == "negative").sum()
    pos_pct  = pos_n / total * 100 if total else 0
    neg_pct  = neg_n / total * 100 if total else 0
    # Net Sentiment Score = % positive minus % negative.
    # A score of +40 means 40% more positive reviews than negative.
    # This is the single most useful executive-level metric in the dashboard.
    net_sent = round(pos_pct - neg_pct, 1)
    net_color = GREEN if net_sent >= 0 else RED

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi("Total Reviews",   fmt_k(total),                         k1)
    kpi("Avg Rating",      f"{avg_rat:.2f}" if avg_rat else "—", k2)
    kpi("Positive",        f"{pos_pct:.1f}%",                    k3, GREEN)
    kpi("Negative",        f"{neg_pct:.1f}%",                    k4, RED)
    kpi("Avg Confidence",  f"{avg_conf*100:.1f}%",               k5)
    kpi("Net Sentiment",   f"{net_sent:+.1f}%",                  k6, net_color)

    divider()

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
                              xaxis_title="Star Rating",
                              yaxis_title="Number of Reviews", **PL)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rating column detected.")

    with r1c2:
        sc      = fdf["predicted_sentiment"].value_counts()
        pos_val = (fdf["predicted_sentiment"] == "positive").sum()
        fig = go.Figure(go.Pie(
            labels=sc.index, values=sc.values, hole=0.6,
            marker_colors=[SENT_COLORS.get(s, "#94A3B8") for s in sc.index],
            textinfo="label+percent", textfont_size=11,
        ))
        fig.add_annotation(
            text=f"<b>{fmt_k(pos_val)}</b><br><span style='font-size:10px'>Positive</span>",
            x=0.5, y=0.5, showarrow=False, font_size=14, font_color=GREEN
        )
        fig.update_layout(title="Sentiment Breakdown", **PL,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        if cols.get("product") and cols["product"] in fdf.columns:
            top_prod = (fdf.groupby(cols["product"])
                          .size().reset_index(name="count")
                          .sort_values("count", ascending=False).head(10))
            top_prod["label"] = top_prod[cols["product"]].apply(truncate_label)
            fig = go.Figure(go.Bar(
                y=top_prod["label"], x=top_prod["count"],
                orientation="h", marker_color=TEAL,
                text=top_prod["count"].apply(fmt_k), textposition="outside",
            ))
            fig.update_layout(title=f"Top {entity_label} by Review Volume",
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
                        x=d[cols["rating"]].astype(str), y=d["count"],
                        marker_color=color,
                        text=d["count"].apply(fmt_k), textposition="outside",
                    ))
            fig.update_layout(barmode="stack", title="Sentiment by Rating",
                              xaxis_title="Star Rating", yaxis_title="Count", **PL)
            st.plotly_chart(fig, use_container_width=True)

    divider()

    # Word frequency
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        if cols.get("text") and cols["text"] in fdf.columns:
            pos_texts = fdf[fdf["predicted_sentiment"] == "positive"][cols["text"]].tolist()
            neg_texts = fdf[fdf["predicted_sentiment"] == "negative"][cols["text"]].tolist()
            pos_freq  = get_word_freq(pos_texts)
            neg_freq  = get_word_freq(neg_texts)

            fig = go.Figure()
            if pos_freq:
                words_p, counts_p = zip(*pos_freq)
                fig.add_trace(go.Bar(name="Positive", y=list(words_p), x=list(counts_p),
                                     orientation="h", marker_color=GREEN))
            if neg_freq:
                words_n, counts_n = zip(*neg_freq)
                fig.add_trace(go.Bar(name="Negative", y=list(words_n), x=list(counts_n),
                                     orientation="h", marker_color=RED))
            fig.update_layout(barmode="group", title="Top Keywords by Sentiment",
                              xaxis_title="Frequency",
                              yaxis=dict(autorange="reversed"), **PL)
            st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        section("Top Positive Reviews")
        top_pos = (fdf[fdf["predicted_sentiment"] == "positive"]
                     .sort_values("confidence_score", ascending=False)
                     .head(5))
        if not top_pos.empty:
            for _, row in top_pos.iterrows():
                text_val     = str(row[cols["text"]]) if cols.get("text") else ""
                text_preview = text_val[:120] + "..." if len(text_val) > 120 else text_val
                prod_label   = truncate_label(row[cols["product"]]) if cols.get("product") and cols["product"] in row else ""
                st.markdown(
                    f'<div class="pos-card">'
                    f'<div class="flag-meta">{prod_label} · {row["confidence_score"]*100:.0f}% confidence</div>'
                    f'<div class="flag-text">{text_preview}</div>'
                    f'</div>', unsafe_allow_html=True,
                )

    divider()

    section("Flagged Negative Reviews")
    flagged = (fdf[fdf["predicted_sentiment"] == "negative"]
                 .sort_values("confidence_score", ascending=False)
                 .head(6))
    if not flagged.empty:
        neg_cols = st.columns(2)
        for i, (_, row) in enumerate(flagged.iterrows()):
            text_val     = str(row[cols["text"]]) if cols.get("text") else ""
            text_preview = text_val[:120] + "..." if len(text_val) > 120 else text_val
            prod_label   = truncate_label(row[cols["product"]]) if cols.get("product") and cols["product"] in row else ""
            with neg_cols[i % 2]:
                st.markdown(
                    f'<div class="flag-card">'
                    f'<div class="flag-meta">{prod_label} · {row["confidence_score"]*100:.0f}% confidence</div>'
                    f'<div class="flag-text">{text_preview}</div>'
                    f'</div>', unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE
# Focuses on entity-level (product/location/etc.) analysis rather than
# individual review analysis. Requires both a product column and ideally a
# rating column — warns and stops if neither is present.
#
# Charts on this page:
#   - Average Rating by Entity (horizontal bar, top 10)
#   - Sentiment by Entity (stacked horizontal bar, top 10)
#   - Top Positive Entities (horizontal bar, top 10)
#   - Top Negative Entities (horizontal bar, top 10)
#   - High-Confidence Negative Heatmap (entity × star rating)
#   - Entity Summary Table (sortable, top 50, downloadable)
# ══════════════════════════════════════════════════════════════════════════════
if page == "Performance":
    if st.session_state.get("df_scored") is None:
        st.info("Upload and score a dataset on the Overview page first.")
        st.stop()

    df   = st.session_state["df_scored"]
    cols = st.session_state.get("cols", {})
    entity_label = st.session_state.get("entity_label", "Product")

    render_filters_panel(df, cols)
    fdf = apply_filters(df, cols)

    section("Product Performance Analysis")
    divider()

    prod_col   = cols.get("product")
    rating_col = cols.get("rating")

    if not prod_col or prod_col not in fdf.columns:
        st.warning("No product column detected. This page requires product data.")
        st.stop()

    fdf["_prod_label"] = fdf[prod_col].apply(truncate_label)

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        if rating_col and rating_col in fdf.columns:
            avg_rat = (fdf.groupby("_prod_label")[rating_col]
                         .mean().reset_index()
                         .sort_values(rating_col, ascending=False).head(10))
            fig = go.Figure(go.Bar(
                y=avg_rat["_prod_label"], x=avg_rat[rating_col].round(2),
                orientation="h", marker_color=GREEN,
                text=avg_rat[rating_col].round(2), textposition="outside",
            ))
            fig.update_layout(title=f"Average Rating by {entity_label}",
                              xaxis_title="Avg Rating",
                              yaxis=dict(autorange="reversed"), **PL)
            st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        sent_prod = (fdf.groupby(["_prod_label", "predicted_sentiment"])
                       .size().reset_index(name="count"))
        top10 = (fdf.groupby("_prod_label").size()
                   .sort_values(ascending=False).head(10).index.tolist())
        sent_prod = sent_prod[sent_prod["_prod_label"].isin(top10)]
        fig = go.Figure()
        for sent, color in SENT_COLORS.items():
            d = sent_prod[sent_prod["predicted_sentiment"] == sent]
            if not d.empty:
                fig.add_trace(go.Bar(
                    name=sent.capitalize(),
                    y=d["_prod_label"], x=d["count"],
                    orientation="h", marker_color=color,
                    text=d["count"].apply(fmt_k), textposition="outside",
                ))
        fig.update_layout(barmode="stack", title=f"Sentiment by {entity_label}",
                          xaxis_title="Review Count",
                          yaxis=dict(autorange="reversed"), **PL)
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        pos_prod = (fdf[fdf["predicted_sentiment"] == "positive"]
                      .groupby("_prod_label").size().reset_index(name="positive_count")
                      .sort_values("positive_count", ascending=False).head(10))
        fig = go.Figure(go.Bar(
            y=pos_prod["_prod_label"], x=pos_prod["positive_count"],
            orientation="h", marker_color=GREEN,
            text=pos_prod["positive_count"].apply(fmt_k), textposition="outside",
        ))
        fig.update_layout(title=f"Top Positive {entity_label}",
                          xaxis_title="Positive Reviews",
                          yaxis=dict(autorange="reversed"), **PL)
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        neg_prod = (fdf[fdf["predicted_sentiment"] == "negative"]
                      .groupby("_prod_label").size().reset_index(name="negative_count")
                      .sort_values("negative_count", ascending=False).head(10))
        fig = go.Figure(go.Bar(
            y=neg_prod["_prod_label"], x=neg_prod["negative_count"],
            orientation="h", marker_color=RED,
            text=neg_prod["negative_count"].apply(fmt_k), textposition="outside",
        ))
        fig.update_layout(title=f"Top Negative {entity_label}",
                          xaxis_title="Negative Reviews",
                          yaxis=dict(autorange="reversed"), **PL)
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    # Shows where high-confidence negative reviews are concentrated — which
    # products are getting the worst reviews at which star ratings.
    # Only includes reviews where confidence >= 85% to keep the signal clean.
    # The pivot table reshapes the data: rows = products, columns = star ratings,
    # values = count of negative reviews. fillna(0) handles missing combinations.
    divider()
    section("High-Confidence Negative Review Heatmap")

    if prod_col and rating_col and prod_col in fdf.columns and rating_col in fdf.columns:
        heat_df = (fdf[(fdf["predicted_sentiment"] == "negative") &
                       (fdf["confidence_score"] >= 0.85)]
                     .groupby(["_prod_label", rating_col])
                     .size().reset_index(name="count"))

        if not heat_df.empty:
            top_neg_prods = (fdf[fdf["predicted_sentiment"] == "negative"]
                               .groupby("_prod_label").size()
                               .sort_values(ascending=False).head(12).index.tolist())
            heat_df = heat_df[heat_df["_prod_label"].isin(top_neg_prods)]
            pivot   = heat_df.pivot(index="_prod_label", columns=rating_col, values="count").fillna(0)

            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[f"{int(c)} Star" for c in pivot.columns],
                y=pivot.index.tolist(),
                colorscale=[[0, "#FFF5F5"], [0.5, "#FCA5A5"], [1, "#EF4444"]],
                text=pivot.values.astype(int),
                texttemplate="%{text}",
                showscale=True,
                colorbar=dict(title="Count"),
            ))
            fig.update_layout(
                title=f"High-Confidence Negatives by {entity_label} and Rating",
                xaxis_title="Star Rating",
                yaxis=dict(autorange="reversed"),
                **PL
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No high-confidence negative reviews in current filter.")
    else:
        st.info("Product and rating columns required for heatmap.")

    divider()
    section("Product Summary Table")

    # ── Summary table ─────────────────────────────────────────────────────────
    # Aggregates all review data down to one row per product/entity.
    # Uses pandas .agg() with named aggregations — each key in agg_dict becomes
    # a column name, and each value is a (source_column, aggregation_function) tuple.
    # Lambda functions let us write custom aggregations like "% positive" inline.
    # The result is sorted by review count and capped at 50 rows for readability.
    agg_dict = {
        "review_count": (prod_col, "count"),
        "pct_positive": ("predicted_sentiment", lambda x: round((x=="positive").sum()/len(x)*100, 1)),
        "pct_negative": ("predicted_sentiment", lambda x: round((x=="negative").sum()/len(x)*100, 1)),
        "avg_confidence": ("confidence_score", "mean"),
    }
    if rating_col and rating_col in fdf.columns:
        agg_dict["avg_rating"] = (rating_col, "mean")

    prod_summary = fdf.groupby("_prod_label").agg(**agg_dict).reset_index()
    prod_summary = prod_summary.sort_values("review_count", ascending=False).head(50)

    col_names = ["Product", "Reviews", "% Positive", "% Negative", "Avg Confidence"]
    if rating_col and rating_col in fdf.columns:
        col_names.append("Avg Rating")
    prod_summary.columns = col_names

    prod_summary["Avg Confidence"] = (prod_summary["Avg Confidence"] * 100).round(1).astype(str) + "%"
    prod_summary["% Positive"]     = prod_summary["% Positive"].astype(str) + "%"
    prod_summary["% Negative"]     = prod_summary["% Negative"].astype(str) + "%"
    if "Avg Rating" in prod_summary.columns:
        prod_summary["Avg Rating"] = prod_summary["Avg Rating"].round(2)

    st.dataframe(prod_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    csv_out = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Dataset", csv_out,
                       file_name="voxiq_filtered.csv", mime="text/csv",
                       use_container_width=True)
