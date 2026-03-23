# app.py — MBIC (Media Bias & Information Classifier)
# Tabs: 📰 Live News (default, auto: technology/popularity/10) | 🔎 Classify | 📊 About / Results
# Local model + NewsAPI via secrets.toml (no sidebar)
# Graceful handling: if NewsAPI fails, show "API offline" banner (no fallback).

import os
import requests
import streamlit as st
import torch
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------------- Page config & global styles ----------------
st.set_page_config(page_title="MBIC — Media Bias Classifier", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px circle at 20% -10%, #0b1220 10%, #0a0f1a 40%, #080d16 80%) fixed;
  color: #e6edf3;
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans";
}
h1,h2,h3,h4 { color: #e6edf3; }

/* Tabs */
.stTabs [data-baseweb="tab-list"]{ justify-content:center; gap:2.5rem; }
.stTabs [data-baseweb="tab"]{ font-size:1.15rem; font-weight:700; padding:0.9rem 1.3rem; border-radius:12px; }
.stTabs [data-baseweb="tab"]:hover{ background:#182235; }

/* Tiles */
.tile{ background:linear-gradient(180deg,#111a2b 0%,#0e1726 100%); border:1px solid #1b2a44;
  border-radius:16px; padding:14px 16px; box-shadow:0 10px 30px rgba(0,0,0,.25);}
.tile .label{ color:#9fb0c9; font-size:.9rem;} .tile .value{ color:#e6edf3; font-size:1.6rem; font-weight:800;}

/* Cards */
.card{ background:linear-gradient(180deg,#0f1828 0%,#0c1422 100%); border:1px solid #1b2a44;
  border-radius:16px; padding:12px; margin-bottom:14px; box-shadow:0 8px 24px rgba(0,0,0,.25);}
.card h4{ margin:0 0 6px 0;} .card .meta{ color:#9fb0c9; font-size:.85rem; }

/* Badges */
.badge{ display:inline-block; padding:6px 10px; border-radius:10px; font-weight:700; margin-top:6px; }
.badge-biased{ background:#3d1f1f; color:#ffb0b0;}
.badge-slight-biased{ background:#2c1f13; color:#ffd4a6;}
.badge-uncertain{ background:#2a2631; color:#d6c9ff;}
.badge-slight-non{ background:#1f2b22; color:#c9f4da;}
.badge-non{ background:#15301f; color:#b6f1c8;}

/* Inputs/buttons */
.stTextInput>div>div>input, textarea, .stSelectbox div[data-baseweb="select"]{
  background:#0e1726 !important; color:#e6edf3 !important; border:1px solid #1b2a44 !important;}
.stSlider>div>div>div>div{ background:#3a6aa6 !important;}
.stButton>button{ background:#1b2a44; border:1px solid #2b4772; color:#e6edf3; font-weight:700;}
.stButton>button:hover{ background:#254066; border-color:#3a6aa6;}

/* Hero */
.hero{ background:linear-gradient(120deg, rgba(42,76,140,.25), rgba(30,54,94,.15));
  border:1px solid #1b2a44; border-radius:18px; padding:18px 20px; margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

# ---------------- Model ----------------
def pick_device():
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return 0  # Apple MPS
    except Exception:
        pass
    return -1  # CPU
MODEL_ID = "stutid04/mbic-distilbert-bias"
@st.cache_resource(show_spinner=True)
def load_classifier():
   tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    model.config.id2label = {0: "Non-biased", 1: "Biased"}
    model.config.label2id = {"Non-biased": 0, "Biased": 1}

    device = pick_device()

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True
    )

    return clf


clf = load_classifier()

# ---------------- Helpers ----------------
def normalize_scores(raw):
    scores = raw[0] if isinstance(raw, list) and len(raw) and isinstance(raw[0], list) else raw
    out = []
    for s in scores:
        lab = s.get("label", "")
        sc = float(s.get("score", 0.0))
        if lab in ("LABEL_0","LABEL_1"):
            lab = "Biased" if lab.endswith("1") else "Non-biased"
        out.append({"label": lab, "score": sc})
    return out

def classify_batch(texts):
    raw = clf(texts)
    return [normalize_scores(r) for r in raw]

def graded_label(p_biased: float):
    if p_biased >= 0.75: return "Biased", "badge-biased"
    if p_biased >= 0.55: return "Slightly biased", "badge-slight-biased"
    if p_biased >  0.45: return "Mixed / uncertain", "badge-uncertain"
    if p_biased >= 0.25: return "Slightly non-biased", "badge-slight-non"
    return "Non-biased", "badge-non"

def gauge_fig(p_biased: float):
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(p_biased),
            number={'suffix': "", 'font': {'color': '#e6edf3'}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 0, 'tickcolor': '#9fb0c9'},
                'bar': {'color': '#7aa2ff'},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0.00, 0.25], 'color': '#15301f'},
                    {'range': [0.25, 0.45], 'color': '#1f2b22'},
                    {'range': [0.45, 0.55], 'color': '#2a2631'},
                    {'range': [0.55, 0.75], 'color': '#2c1f13'},
                    {'range': [0.75, 1.00], 'color': '#3d1f1f'},
                ],
            },
            domain={'x': [0, 1], 'y': [0, 1]}
        )
    ).update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6edf3'),
        height=140,
    )

def stat_tile(label: str, value: str):
    st.markdown(f"<div class='tile'><div class='label'>{label}</div><div class='value'>{value}</div></div>",
                unsafe_allow_html=True)

# ---- NewsAPI fetch with graceful offline banner
def fetch_news(q, sort_by="publishedAt", page_size=10):
    """Return (articles, status). status ∈ {'ok','offline'}."""
    key = os.getenv("NEWSAPI_KEY") or st.secrets.get("NEWSAPI_KEY", "")
    if not key:
        return [], "offline"
    try:
        url = "https://newsapi.org/v2/everything"
        params = dict(q=q, language="en", sortBy=sort_by, pageSize=page_size, apiKey=key)
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("articles", []), "ok"
    except requests.exceptions.RequestException:
        return [], "offline"

# ---------------- Hero ----------------
st.markdown("""
<div class='hero'>
  <h2 style='margin:0;'>🧭 MBIC — Media Bias & Information Classifier</h2>
  <div style='color:#9fb0c9;'>DistilBERT (local) • English headlines • Bias spectrum with gauges</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab_live, tab_classify, tab_about = st.tabs(["📰 Live News", "🔎 Classify", "📊 About / Results"])

# ====== Tab 1: Live News (auto: technology/popularity/10) ======
with tab_live:
    default_topic, default_sort, default_articles = "technology", "popularity", 10

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1: query = st.text_input("Search topic", default_topic)
    with c2: sort_by = st.selectbox("Sort by", ["publishedAt", "relevancy", "popularity"], index=2)
    with c3: page_size = st.selectbox("Articles", [5, 10, 15], index=1)

    # Autorun once with defaults
    if "auto_loaded" not in st.session_state:
        st.session_state["auto_loaded"] = True
        trigger = True
        query, sort_by, page_size = default_topic, default_sort, default_articles
    else:
        trigger = False

    if st.button("Fetch & Classify", use_container_width=True) or trigger:
        articles, status = fetch_news(query, sort_by, page_size)

        if status != "ok":
            st.error("🛰 News API offline / unreachable. Please check your network or try again later.")
        if not articles:
            st.info("No articles to display right now.")
        else:
            texts = [
                (" ".join([a.get("title") or "", a.get("description") or ""]).strip()) or (a.get("title") or "(no text)")
                for a in articles
            ]
            with st.spinner("🧠 Classifying headlines..."):
                all_scores = classify_batch(texts)

            # Stats
            biased_cnt = 0
            avg_biased = 0.0
            for sc in all_scores:
                mapped = {s["label"]: s["score"] for s in sc}
                p_b = mapped.get("Biased", 0.0)
                if p_b >= 0.5: biased_cnt += 1
                avg_biased += p_b
            avg_biased = avg_biased / max(len(all_scores), 1)

            t1, t2, t3 = st.columns(3)
            with t1: stat_tile("Articles Analyzed", f"{len(all_scores)}")
            with t2: stat_tile("Predicted Biased (≥0.5)", f"{biased_cnt}")
            with t3: stat_tile("Avg P(Biased)", f"{avg_biased:.2f}")
            st.write("")

            # Cards with graded label + gauge
            for idx, a in enumerate(articles):
                title = a.get("title") or "(no title)"
                desc = a.get("description") or ""
                url = a.get("url")
                src = (a.get("source") or {}).get("name", "")
                published = (a.get("publishedAt") or "").replace("T"," ").replace("Z"," UTC")
                img = a.get("urlToImage")

                scores = all_scores[idx] if idx < len(all_scores) else []
                s_map = {s["label"]: s["score"] for s in scores}
                p_biased = float(s_map.get("Biased", 0.0))
                p_non = float(s_map.get("Non-biased", 0.0))
                grade, badge_class = graded_label(p_biased)

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                colA, colB = st.columns([1.1, 2.6], vertical_alignment="center")
                with colA:
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='width:100%;aspect-ratio:16/9;background:#0b1220;"
                            "border:1px solid #1b2a44;border-radius:12px;display:flex;align-items:center;"
                            "justify-content:center;color:#9fb0c9'>No image</div>",
                            unsafe_allow_html=True,
                        )
                    st.plotly_chart(gauge_fig(p_biased), use_container_width=True, config={"displayModeBar": False})
                with colB:
                    st.markdown(f"<h4>{title}</h4>", unsafe_allow_html=True)
                    meta = " • ".join([x for x in [src, published] if x])
                    if meta: st.markdown(f"<div class='meta'>{meta}</div>", unsafe_allow_html=True)
                    if desc: st.write(desc)
                    if url: st.markdown(f"[🔗 Read full article]({url})")
                    st.markdown(f"<span class='badge {badge_class}'>{grade}</span>", unsafe_allow_html=True)
                    st.caption(f"P(Non-biased)={p_non:.2f} • P(Biased)={p_biased:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

# ====== Tab 2: Classify ======
with tab_classify:
    st.subheader("Classify Custom Text")
    t1, t2 = st.columns([3, 1])
    with t1:
        text_input = st.text_area(
            "Paste your headline or paragraph 👇",
            height=150,
            placeholder="Example: Government accused of hiding key evidence in climate report",
        )
    with t2:
        threshold_cls = st.slider("Binary threshold", 0.0, 1.0, 0.50, 0.01, key="thr_cls",
                                  help="Only used for binary label; gauge still shows graded spectrum.")
    if st.button("Classify Text", type="primary", key="btn_cls") and text_input.strip():
        with st.spinner("Analyzing..."):
            res = clf(text_input.strip())
            scores = normalize_scores(res)
            scores = sorted(scores, key=lambda x: x["score"], reverse=True)
            p_b = next((s["score"] for s in scores if s["label"]=="Biased"), 0.0)
            p_non = next((s["score"] for s in scores if s["label"]=="Non-biased"), 0.0)
            grade, badge_class = graded_label(p_b)
            binary = "Biased" if p_b >= threshold_cls else "Non-biased"

            left, right = st.columns([1,2])
            with left:
                stat_tile("Binary (thresholded)", binary)
                st.markdown(f"<span class='badge {badge_class}'>{grade}</span>", unsafe_allow_html=True)
                st.caption(f"P(Non-biased)={p_non:.2f} • P(Biased)={p_b:.2f}")
            with right:
                st.plotly_chart(gauge_fig(p_b), use_container_width=True, config={"displayModeBar": False})
            st.json([{s['label']: round(s['score'], 4)} for s in scores])
    elif st.button("Classify Text", key="btn_cls_disabled"):
        st.warning("Please paste some text.")

# ====== Tab 3: About / Results ======
with tab_about:
    st.title("📊 MBIC — Model Overview & Results")
    st.caption("Machine Bias & Information Classification | Local DistilBERT Model")

    st.header("1️⃣ Motivation & Objective")
    st.markdown("""
The **MBIC (Media Bias & Information Classifier)** project detects *biased language patterns* in short texts:
- News headlines  
- Social media posts  
- Short editorial snippets  

Goal: **quantify potential linguistic bias** — whether statements lean toward emotionally charged framing or remain neutral.
""")

    st.header("2️⃣ Model Architecture")
    st.markdown("""
**Model Base:** 🧠 `DistilBERT-base-uncased`  
**Fine-tuned On:** `labeled_dataset.xlsx` with labels:
- `Non-biased` → 0  
- `Biased` → 1  

**Summary**
- Tokenizer: `AutoTokenizer` (DistilBERT)
- Encoder: 6-layer DistilBERT
- Head: linear (2 logits) + softmax
- Loss: CrossEntropy
- Optimizer: AdamW
- Frameworks: 🤗 Transformers + PyTorch + Evaluate
""")
    st.markdown("**🧩 Flow:** *Embedding → Transformer → Classifier*")

    st.header("3️⃣ Dataset & Labeling")
    st.markdown("""
**Data Source:** Curated manually + public headlines  
- ~1,500 samples  
- Split: 80% train / 20% test  
- Balanced after cleaning
""")

    # Arrow-safe table (Counts as strings)
    overview_rows = [
        ("Total Samples", "1551"),
        ("Train Set", "1240"),
        ("Test Set", "311"),
        ("Labels", "2 (Biased / Non-biased)"),
    ]
    overview_df = pd.DataFrame(overview_rows, columns=["Metric", "Count"]).astype({"Count": "string"})
    st.table(overview_df)

    st.header("4️⃣ Training Configuration")
    st.code("""Model: distilbert-base-uncased
Batch size (train): 16
Batch size (eval): 32
Learning rate: 2e-5
Epochs: 4
Weight decay: 0.01
Optimizer: AdamW
Device: Apple MPS (Metal GPU)
""", language="yaml")

    st.header("5️⃣ Evaluation Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "73.6%")
    c2.metric("F1 (macro)", "0.68")
    c3.metric("F1 (weighted)", "0.72")

    metrics_df = pd.DataFrame({"Metric":["Accuracy","F1 (macro)","F1 (weighted)"], "Score":[0.736,0.68,0.72]})
    chart_metrics = (
        alt.Chart(metrics_df).mark_bar().encode(
            x=alt.X("Metric:N", sort=None, axis=alt.Axis(labelColor="#cfd8e3", title=None)),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format="~p", labelColor="#cfd8e3")),
            tooltip=["Metric","Score"]
        ).properties(height=220)
        + alt.Chart(metrics_df).mark_text(dy=-6, fontWeight="bold").encode(
            x="Metric:N", y="Score:Q", text=alt.Text("Score:Q", format=".2f")
        )
    )
    st.altair_chart(chart_metrics, use_container_width=True)

    st.markdown("""
Results are on a 311-sample held-out test set.  
The model shows decent separation — it **learns bias cues** like subjective framing and emotionally loaded verbs.
""")

    st.header("6️⃣ Sample Model Predictions")
    samples = [
        ("Government accused of covering up data", "Biased"),
        ("China to remove tariffs on US agriculture goods from Nov 10", "Non-biased"),
        ("New reforms spark outrage across citizens", "Biased"),
        ("Prime Minister to address nation tomorrow", "Non-biased"),
    ]
    for text, label in samples:
        color = "#c0392b" if label == "Biased" else "#27ae60"
        st.markdown(
            f"<div style='padding:10px;border-radius:10px;background:{color}22;margin:5px 0;'>"
            f"<b>{text}</b><br><i>Predicted: {label}</i></div>",
            unsafe_allow_html=True,
        )

    st.header("7️⃣ Limitations & Future Work")
    st.markdown("""
- 🧩 Small dataset → possible topic overfit  
- 🌍 Monolingual (EN) only  
- 🧮 Next: cross-lingual fine-tuning (XLM-R / mMiniLM)  
- 💬 Add sentiment & stance for richer explanations
""")

    st.header("8️⃣ Tools & Frameworks")
    tools_df = pd.DataFrame({
        "Component": ["NLP Backbone","ML Framework","UI","Dataset","Deployment"],
        "Tool/Library": ["🤗 Transformers (DistilBERT)","PyTorch","Streamlit","Pandas + Excel","Streamlit Cloud"],
        "Purpose": ["Bias detection","Model training","Interactive dashboard","Data handling","Lightweight hosting"]
    }).astype({"Component":"string","Tool/Library":"string","Purpose":"string"})
    st.table(tools_df)

    st.markdown("### 📊 Dataset Label Distribution (Proportions, max=1)")
    try:
        if os.path.exists("labeled_dataset.xlsx"):
            raw_df = pd.read_excel("labeled_dataset.xlsx")
            raw_df = raw_df[["sentence","Label_bias"]].dropna()
            label_map = {"Non-biased": 0, "Biased": 1}
            raw_df = raw_df[raw_df["Label_bias"].isin(label_map)]
            counts = raw_df["Label_bias"].value_counts().reindex(["Non-biased","Biased"]).fillna(0)
            props = (counts / counts.sum()).reset_index()
            props.columns = ["Label", "Proportion"]
            chart_dist = (
                alt.Chart(props).mark_bar().encode(
                    x=alt.X("Label:N", sort=None, axis=alt.Axis(labelColor="#cfd8e3", title=None)),
                    y=alt.Y("Proportion:Q", scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format="~p", labelColor="#cfd8e3")),
                    tooltip=["Label","Proportion"]
                ).properties(height=220)
                + alt.Chart(props).mark_text(dy=-6, fontWeight="bold").encode(
                    x="Label:N", y="Proportion:Q", text=alt.Text("Proportion:Q", format=".2f")
                )
            )
            st.altair_chart(chart_dist, use_container_width=True)
        else:
            st.info("`labeled_dataset.xlsx` not found in the app directory.")
    except Exception as e:
        st.warning(f"Could not read dataset for visualization: {e}")

    st.markdown("---")
    st.caption("Built with Streamlit + Transformers • Local DistilBERT bias classifier • Live headlines via NewsAPI")
