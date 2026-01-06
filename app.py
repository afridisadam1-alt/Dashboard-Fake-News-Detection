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
import tensorflow as tf


import sklearn
st.write(sklearn.__version__)

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
# Dataset configs (local files only)
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
# Load dataset (local only)
# =========================================================
@st.cache_data(show_spinner=False)
def load_dataset(cfg, dataset_name):
    csv_file = cfg["csv"]
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
# Load ML model (local only)
# =========================================================
# =========================================================
# Load ML model (Streamlit Cloud safe)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_ml_model(model_file, vector_file):
    import pickle
    from pathlib import Path

    model_path = Path(__file__).parent / model_file
    vector_path = Path(__file__).parent / vector_file

    # Check if files exist
    if not model_path.exists():
        st.error(f"ML model file not found: {model_path}")
        return None, None
    if not vector_path.exists():
        st.error(f"Vectorizer file not found: {vector_path}")
        return None, None

    # Load pre-fitted model and vectorizer
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vector_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Debug check
    if not hasattr(vectorizer, "idf_"):
        st.error(f"Vectorizer {vector_file} is not fitted properly!")
    
    return model, vectorizer


    # Check if vectorizer is fitted
    if not hasattr(vec, "idf_"):
        st.warning(f"Vectorizer {vector_file} is not fitted. Fitting on dataset text...")
        if df_text is not None:
            vec.fit(df_text)
            st.success(f"Vectorizer {vector_file} fitted automatically.")
        else:
            st.error(f"Cannot fit vectorizer: dataset text not provided.")
    
    return model, vec

# =========================================================
# Load DL model (local only)
# =========================================================
@st.cache_resource
def load_dl_model(model_file, tokenizer_file, dataset_name):
    model = load_model(model_file, compile=False)
    with open(tokenizer_file, "rb") as f: tok = pickle.load(f)
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
# Sidebar – model selection
st.sidebar.title("Model Selection")
model_type = st.sidebar.radio("Select Model Type:", ["ML (Traditional)", "DL (BiLSTM)"])

if model_type == "ML (Traditional)":
    dataset = st.sidebar.selectbox("Select Dataset", list(ml_datasets))
    cfg = ml_datasets[dataset]
    
    # Load ML model and vectorizer (Streamlit Cloud safe)
    model, vectorizer = load_ml_model(cfg["model"], cfg["vectorizer"])
    if model is None or vectorizer is None:
        st.stop()  # Stop the app if files are missing

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
df_vc = df[df[label_col].notna()]
st.dataframe(df_vc[label_col].value_counts().rename("Count"), width='stretch')

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
    width='stretch'
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
            st.altair_chart(chart)
        else:
            st.info("No prominent words detected.")
    else:
        st.subheader("☁️ Word Cloud (Deep Learning Input Text)")
        wc = WordCloud(width=800,height=400,background_color="white",colormap="viridis").generate(st.session_state.input_text)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc.to_image())
        ax.axis("off")
        st.pyplot(fig)
