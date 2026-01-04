# =========================================================
# app.py – Disinformation Detection Dashboard (DL + ML Cached, Toolbar Hidden)
# =========================================================

import os, warnings, pickle, re
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gdown
import tensorflow as tf

# =========================================================
# Check package versions
# =========================================================
print("streamlit:", st.__version__)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("PyPDF2:", PdfReader.__module__.split('.')[0], "version not directly accessible")
print("python-docx:", Document.__module__.split('.')[0], "version not directly accessible")
print("tensorflow:", tf.__version__)
print("altair:", alt.__version__)
print("wordcloud:", WordCloud.__module__.split('.')[0], "version not directly accessible")
print("gdown:", gdown.__version__)

# =========================================================
# System cleanup
# =========================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Disinformation Detection Dashboard", layout="wide")

# =========================================================
# Constants
# =========================================================
MAX_LEN = 700

# =========================================================
# Dataset configs
# =========================================================
ml_datasets = {
    "George McIntire": {"model": "pandy_pac.pkl", "vectorizer": "pandy_vectorizer.pkl", "csv": "Pandy-Dataset_sample20.csv", "text_col": "text"},
    "EUvsIPF": {"model": "euvsipf_pac.pkl", "vectorizer": "euvsipf_vectorizer.pkl", "csv": "EUvsIPF-Dataset_sample20.csv", "text_col": "text_english"},
    "EUvsDisinfo": {"model": "euvsdisinfo_pac.pkl", "vectorizer": "euvsdisinfo_vectorizer.pkl", "csv": "EUvsDisinfo-Dataset_sample20.csv", "text_col": "text_english"},
}

dl_datasets = {
    "FA-KES": {"model": "FA-KES_bilstm.h5", "tokenizer": "FA-KES_tokenizer.pkl", "csv": "FA-KES-Dataset_sample20.csv", "text_col": "text"},
    "ISOT": {"model": "ISOT_bilstm.h5", "tokenizer": "ISOT_tokenizer.pkl", "csv": "ISOT-Dataset_sample20.csv", "text_col": "text"},
    "EUvsISOT": {"model": "EUvsISOT_bilstm.h5", "tokenizer": "EUvsISOT_tokenizer.pkl", "csv": "EUvsISOT-Dataset_sample20.csv", "text_col": "text_english"},
}

# =========================================================
# Prediction label mapping
# =========================================================
PRED_LABEL_MAP = {
    "FA-KES": {0: "TRUE", 1: "Fake"},
    "ISOT": {0: "True", 1: "Fake"},
    "EUvsISOT": {0: "True", 1: "disinformation"},
    "George McIntire": {0: "REAL", 1: "FAKE"},
    "EUvsIPF": {0: "true", 1: "disinformation"},
    "EUvsDisinfo": {0: "support", 1: "disinformation"},
}

DL_POSITIVE_LABEL = {
    "FA-KES": PRED_LABEL_MAP["FA-KES"],
    "ISOT": PRED_LABEL_MAP["ISOT"],
    "EUvsISOT": PRED_LABEL_MAP["EUvsISOT"],
}

# =========================================================
# Label normalization
# =========================================================
def normalize_prediction(dataset_name, pred_label):
    map_ = PRED_LABEL_MAP.get(dataset_name, {})
    if isinstance(pred_label, str) and pred_label.isdigit():
        pred_label = int(pred_label)
    mapped = map_.get(pred_label, pred_label)
    mapped_lower = str(mapped).strip().lower()
    if mapped_lower in ["true", "real", "support", "0"]:
        return "True"
    elif mapped_lower in ["fake", "1"]:
        return "Fake"
    elif mapped_lower in ["disinformation"]:
        return "Disinformation"
    return str(mapped)

# =========================================================
# Detect label column
# =========================================================
def detect_label_column(df, dataset_name):
    if dataset_name == "FA-KES": return "labels"
    elif dataset_name == "ISOT": return "label"
    elif dataset_name == "EUvsISOT": return "class"
    else:
        for col in ["class", "label", "labels"]:
            if col in df.columns: return col
    return None

# =========================================================
# Google Drive IDs
# =========================================================
GDRIVE_SAMPLE_FILES = {
    "George McIntire": "1RXnaRJfTUnuYgK1Pi37qUqbhaUtMbii9",
    "EUvsIPF": "1n0ZBYhWecGo_3ZlvgUcR0TOOLoNG41vC",
    "EUvsDisinfo": "1u4cpDTEJ-3L7jF5far0IK9kkf7PGZKea",
    "FA-KES": "1xYHjkuNGhlmUy1iOMqRXfMXUOwACiSvb",
    "ISOT": "17FCbvF2A12iE6TlHcbT4a5jrHhvIYT8t",
    "EUvsISOT": "1T3COBdd6Co9OnCSwh7_ehOsVVJePGeP-",
}

GDRIVE_ML_MODELS = {
    "George McIntire": {"model": "1eoEhR0P-WXOXVGgMBIxiLB2XEit9D6fQ", "vectorizer": "1Im-O9ARU9DntAiHdfN8ZsKwwS3SPrZPM"},
    "EUvsIPF": {"model": "1lDV23rhEL5X9bkBKCoQB-ShjwtOSN8pX", "vectorizer": "1k7oAtqcD7AJ1WwmGpmHMsHQ3zjC2OxlT"},
    "EUvsDisinfo": {"model": "1SnfQVPoSM9SiNgOfJ7AoYdDZEjySzmxS", "vectorizer": "1lFMmiigHiDbZi70u1Rcb2-bIpQxkSPsD"},
}

GDRIVE_DL_MODELS = {
    "FA-KES": {"model": "1e_2oC7IuT3GMW4pboqvD08bNh1xomuPK", "tokenizer": "1xUs4rWebBncfPl25w0CWMF3im0qSD8Ki"},
    "ISOT": {"model": "1ir5P0koTRvPtdT28wim6ZbS5kRnziUGa", "tokenizer": "1qT3ltg06aBX3TJBfFBW8xFeHuTPdgLLq"},
    "EUvsISOT": {"model": "15dHJ-jwycNppM9QaKNWsyrRb5f4wPB2X", "tokenizer": "13UyrLbmYXmL6FzyRhOnmR2ipBBJ_0Mwi"},
}

# =========================================================
# Download from Google Drive
# =========================================================
def download_from_gdrive(file_id, filename):
    if not Path(filename).exists() or Path(filename).stat().st_size == 0:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    return filename

# =========================================================
# Load dataset (cached)
# =========================================================
@st.cache_data(show_spinner=False)
def load_dataset(cfg, dataset_name):
    file_id = GDRIVE_SAMPLE_FILES[dataset_name]
    csv_file = download_from_gdrive(file_id, cfg["csv"])
    df = pd.read_csv(csv_file)
    label_col = detect_label_column(df, dataset_name)
    text_col = cfg["text_col"]
    df[label_col] = pd.Series(df[label_col], dtype=str).str.strip()
    df[text_col] = pd.Series(df[text_col], dtype=str).str.strip()
    df = df[~df[label_col].isin(["", "nan", "NaN"])]
    df = df[df[text_col] != ""]
    df = df[[text_col, label_col]].copy()
    df["Select"] = False
    return df, label_col

# =========================================================
# Load ML model (cached)
# =========================================================
@st.cache_resource
def load_ml_model(m, v, dataset_name):
    model_file = download_from_gdrive(GDRIVE_ML_MODELS[dataset_name]["model"], m)
    vec_file = download_from_gdrive(GDRIVE_ML_MODELS[dataset_name]["vectorizer"], v)
    with open(model_file, "rb") as f: model = pickle.load(f)
    with open(vec_file, "rb") as f: vec = pickle.load(f)
    return model, vec

# =========================================================
# Load DL model (cached)
# =========================================================
@st.cache_resource
def load_dl_model(m, t, dataset_name):
    model_file = download_from_gdrive(GDRIVE_DL_MODELS[dataset_name]["model"], m)
    tok_file = download_from_gdrive(GDRIVE_DL_MODELS[dataset_name]["tokenizer"], t)
    model = load_model(model_file, compile=False)
    with open(tok_file, "rb") as f: tok = pickle.load(f)
    return model, tok

# =========================================================
# Hide download button / toolbar
# =========================================================
st.markdown("""
    <style>
        [data-testid="stElementToolbar"] {display: none !important;}
        [data-testid="stDataFrameToolbar"] {display: none !important;}
    </style>
""", unsafe_allow_html=True)

# =========================================================
# Sidebar – model selection
# =========================================================
st.sidebar.title("Model Selection")
model_type = st.sidebar.radio("Select Model Type:", ["ML (Traditional)", "DL (BiLSTM)"])
if model_type == "ML (Traditional)":
    dataset = st.sidebar.selectbox("Select Dataset", list(ml_datasets))
    cfg = ml_datasets[dataset]
    model, vectorizer = load_ml_model(cfg["model"], cfg["vectorizer"], dataset)
    is_ml = True
else:
    dataset = st.sidebar.selectbox("Select Dataset", list(dl_datasets))
    cfg = dl_datasets[dataset]
    model, tokenizer = load_dl_model(cfg["model"], cfg["tokenizer"], dataset)
    is_ml = False

df, label_col = load_dataset(cfg, dataset)
text_col = cfg["text_col"]

# =========================================================
# Prediction function
# =========================================================
def predict(text):
    clean = re.sub(r"[^\x00-\x7F]+", " ", text).strip()
    if is_ml:
        pred = model.predict(vectorizer.transform([clean.lower()]))[0]
        return normalize_prediction(dataset, pred)
    else:
        seq = tokenizer.texts_to_sequences([clean])
        X = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        pred_prob = float(model.predict(X, verbose=0)[0][0])
        pred_class = 1 if pred_prob > 0.5 else 0
        label_map = DL_POSITIVE_LABEL.get(dataset, PRED_LABEL_MAP.get(dataset, {}))
        return label_map.get(pred_class, "Disinformation" if pred_class==1 else "True")

# =========================================================
# Dataset exploration
# =========================================================
st.subheader("Dataset Label Distribution")
df_vc = df[df[label_col].notna()]  # FIXED: define df_vc
st.dataframe(df_vc[label_col].value_counts().rename("Count"), width='stretch')  # Corrected

# =========================================================
# Text search / filter
# =========================================================
valid_labels = [l for l in df[label_col].unique() if str(l).strip().lower() not in ["","nan"]]
labels = ["All"] + sorted(valid_labels)
label_filter = st.radio("Filter by label:", labels, horizontal=True)
df_f = df if label_filter=="All" else df[df[label_col]==label_filter]
df_f = df_f[df_f[label_col].notna()]
df_f.reset_index(drop=True, inplace=True)

search_query = st.text_input("Search in text column:")
if search_query:
    df_f = df_f[df_f[text_col].str.contains(search_query, case=False, na=False)]
    df_f.reset_index(drop=True, inplace=True)

# =========================================================
# Dataset view / data editor
# =========================================================
df_view = df_f.groupby(label_col, group_keys=False).head(10) if label_filter=="All" else df_f.head(20)
edited = st.data_editor(
    df_view,
    hide_index=True,
    disabled=[c for c in df_view.columns if c!="Select"],
    width='stretch'  # Corrected
)

selected_rows = edited[edited["Select"]==True]
if not selected_rows.empty:
    st.session_state.input_text = selected_rows.iloc[0][text_col]

# =========================================================
# File uploader
# =========================================================
uploaded_file = st.file_uploader("Upload a file (txt, pdf, docx, csv, xlsx)", type=["txt","pdf","docx","csv","xlsx"])
if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    try:
        if ext=="pdf":
            reader=PdfReader(uploaded_file)
            st.session_state.input_text="\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        elif ext=="docx":
            doc=Document(uploaded_file)
            st.session_state.input_text="\n".join(p.text for p in doc.paragraphs)
        elif ext in ["txt","csv"]:
            st.session_state.input_text = uploaded_file.read().decode("utf-8",errors="ignore")
        elif ext=="xlsx":
            df_file = pd.read_excel(uploaded_file)
            text_cols = [c for c in df_file.columns if df_file[c].dtype==object]
            st.session_state.input_text = "\n".join(df_file[col].astype(str).str.cat(sep="\n") for col in text_cols)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# =========================================================
# Text area
# =========================================================
st.session_state.input_text = st.text_area("Enter text to predict:", st.session_state.get("input_text",""), height=200)

# =========================================================
# Prediction button
# =========================================================
if st.button("Predict") and st.session_state.input_text:
    label = predict(st.session_state.input_text)
    st.success(f"Prediction ({dataset}): {label}")

    if is_ml:
        st.subheader("📊 Most Important Words (Unigram + Bigram TF-IDF)")
        X = vectorizer.transform([st.session_state.input_text.lower()])
        if X.nnz > 0:
            names = vectorizer.get_feature_names_out()
            pw = pd.DataFrame({"Word":names[X.indices],"Importance":X.data}).sort_values("Importance",ascending=False).head(20)
            chart = alt.Chart(pw).mark_bar().encode(
                x=alt.X("Importance:Q"),
                y=alt.Y("Word:N", sort='-x'),
                tooltip=["Word","Importance"]
            ).properties(height=400)
            st.altair_chart(chart, width='container')  # Keep 'container' for charts
        else:
            st.info("No prominent words detected.")
    else:
        st.subheader("☁️ Word Cloud (Deep Learning Input Text)")
        wc = WordCloud(width=800,height=400,background_color="white",colormap="viridis").generate(st.session_state.input_text)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc.to_image())
        ax.axis("off")
        st.pyplot(fig)
