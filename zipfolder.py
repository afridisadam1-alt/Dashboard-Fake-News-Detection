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

#Gemini API
import google.generativeai as genai
import streamlit as st

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
except Exception as e:
    st.error("Gemini API not available. Please check API key.")
    gemini_model = None
# =========================================================
# SYSTEM INTRO (HERO SECTION)
# =========================================================
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 10px;
}

.sub-text {
    font-size: 18px;
    text-align: center;
    color: #444;
    margin-bottom: 25px;
    line-height: 1.6;
}

.author-box {
    font-size: 16px;
    text-align: center;
    color: #666;
    margin-top: 10px;
}
            /* Expander title (this is the clickable header) */
[data-testid="stExpander"] summary {
    font-size: 22px !important;
    font-weight: 700;
    color: #1f4e79;
}
</style>

<div class="big-title">
🧠 AI-Based Disinformation Detection System
</div>

<div class="sub-text">
A research-driven platform combining Machine Learning (ML) and Deep Learning (BiLSTM) 
to detect whether online news content is <b>True</b> or <b>Disinformation</b>.<br><br>

The system integrates text preprocessing, predictive modeling, and explainable AI visualizations 
to support transparent and interpretable decision-making.
</div>

<div class="author-box">
<b>Developed by:</b> Mr Sadam Hussain (PhD Sholar) | Disinformation Detection & AI Systems<br>
<b>Purpose:</b> Academic research on fake news detection, user trust, and human-AI interaction
</div>
            
""", unsafe_allow_html=True)
with st.expander("📘 User Testing Instructions", expanded=False):
    st.markdown("""
    ## Welcome to the AI-Based Disinformation Detection Dashboard 

    ### Testing Tasks
    1. Select ML or DL model (Top left sidebar)
    2. Search dataset content (From text box below or click readio buttons below)
    3. Upload a custom file (with Browse Files) or select a row from the dataset view using radio buttons)
    4. Click Predict (after selecting above records)
    5. Review explanation visualizations
    6. Understand the predicted label (True vs Disinformation) and AI explanation
    7. Complete feedback questionnaire

    ### Prediction Labels
    - True = reliable content
    - Disinformation = misleading content
    """)
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
# Text cleaning function
# =========================================================
def clean_text(text):
    """
    Removes special/garbled characters and extra whitespace
    """
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    text = re.sub(r"\s+", " ", text)            # collapse multiple spaces
    return text.strip()

# =========================================================
# Dataset configs (local files only)
# =========================================================
ml_datasets = {
"EUvsIGF": {"model": "euvsipf_pac.pkl", "vectorizer": "euvsipf_vectorizer.pkl", "csv": "EUvsIPF-Dataset_sample20.csv", "text_col": "text_english"},
}

dl_datasets = {

    "EUvsIGF": {"model": "EUvsIPF_bilstm.h5", "tokenizer": "EUvsIPF_tokenizer.pkl", "csv": "EUvsIPF-Dataset_sample20.csv", "text_col": "text_english"},
}

# =========================================================
# Prediction label mapping
# =========================================================
PRED_LABEL_MAP = {
    "FA-KES": {0: "TRUE", 1: "Fake"},
    "ISOT": {0: "True", 1: "Fake"},
    "EUvsISOT": {0: "True", 1: "disinformation"},
    "George McIntire": {0: "REAL", 1: "FAKE"},
    "EUvsIGF": {0: "true", 1: "disinformation"},
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
    df[text_col] = df[text_col].apply(clean_text)  # <-- clean dataset text
    df = df[~df[label_col].isin(["", "nan", "NaN"])]
    df = df[df[text_col] != ""]
    df = df[[text_col, label_col]].copy()
    df["Select"] = False
    return df, label_col

# =========================================================
# Load ML model (local only)
# =========================================================
@st.cache_resource
def load_ml_model(model_file, vector_file, dataset_name):
    with open(model_file, "rb") as f: model = pickle.load(f)
    with open(vector_file, "rb") as f: vec = pickle.load(f)
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
    clean = clean_text(text)  # <-- clean input text
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
df_view = df_view.reset_index(drop=True)

# ---------------------------------------------------------
# 1. Show dataset (read-only table)
# ---------------------------------------------------------


# ---------------------------------------------------------
# 2. Radio selection (TRUE SINGLE CHOICE)
# ---------------------------------------------------------
st.subheader("Select a record for prediction")

options = df_view.index.tolist()

selected_index = st.radio(
    "Choose row index:",
    options,
    format_func=lambda x: df_view.loc[x, text_col][:180]  # preview text
)

# ---------------------------------------------------------
# 3. Store selected text
# ---------------------------------------------------------
st.session_state.input_text = df_view.loc[selected_index, text_col]
# =========================================================
# File uploader
# =========================================================
import zipfile
import io
# =========================================================
# ZIP Folder Upload + Internal File Browser
# =========================================================

import zipfile
import io

import zipfile
import io

ZIP_PATH = "news_data.zip"   # 👈 your file name

try:
    zip_data = zipfile.ZipFile(ZIP_PATH)

    file_list = [
        f for f in zip_data.namelist()
        if (
            f.endswith((".txt", ".pdf", ".docx", ".csv"))
            and "__MACOSX" not in f
            and not f.startswith(".")
        )
    ]

    st.subheader("📁 Dataset Files (Auto Loaded)")

    selected_file = st.selectbox(
        "Select a file for prediction:",
        file_list
    )

    if selected_file:

        file_bytes = zip_data.read(selected_file)
        ext = selected_file.split(".")[-1].lower()

        text = ""

        if ext == "txt":
            text = file_bytes.decode("utf-8", errors="ignore")

        elif ext == "pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)

        elif ext == "docx":
            doc = Document(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in doc.paragraphs)

        elif ext == "csv":
            text = file_bytes.decode("utf-8", errors="ignore")

        st.session_state.input_text = clean_text(text)

        st.success(f"Loaded: {selected_file}")

except FileNotFoundError:
    st.error("news_data.zip not found. Please place it in the same folder as app.py")

# =========================================================
# Text area
# =========================================================
# =========================================================
# Display selected or uploaded text (manual input disabled)
# =========================================================
if "input_text" in st.session_state and st.session_state.input_text.strip() != "":
    st.text_area(
        "Selected text for prediction (cannot edit):",
        st.session_state.input_text,
        height=200,
        disabled=True
    )
else:
    st.info("Select a row from the dataset or upload a file to predict")
#AI summary function
@st.cache_data(show_spinner=False)
def generate_summary(keywords, label, model_type, dataset):
    try:
        prompt = f"""
You are an explainable AI assistant for fake news detection.

Dataset: {dataset}
Model Type: {model_type}
Predicted Label: {label}

Important Keywords Used By Model:
{keywords}

Explain in simple language:
1. Why keywords influenced prediction
2. Why label = {label}
3. Keep under 100 words
"""

        response = gemini_model.generate_content(prompt)
        return response.text

    except Exception as e:
        return "AI explanation unavailable (quota or API error)."
# =========================================================
# Dynamic color Predict button
# =========================================================
button_color = "#4CAF50"  # default green
button_placeholder = st.empty()

# Only allow prediction if input_text exists
if "input_text" in st.session_state and st.session_state.input_text.strip() != "":
    if button_placeholder.button("Predict", key="predict_btn"):

        # ================= PREDICTION =================
        label = predict(st.session_state.input_text)
        # ================= RESULT DISPLAY CARD =================
        label_color = "#FF4B4B" if label.lower() in ["fake", "disinformation"] else "#4CAF50"
        label_icon = "🚨" if label.lower() in ["fake", "disinformation"] else "✅"

        st.markdown(f"""
        <div style="
            background-color: {label_color};
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: 400;
            color: white;
            margin-top: 15px;
            animation: fadeIn 0.8s ease-in-out;
        ">
            {label_icon} {label.upper()}
        </div>

        <style>
        @keyframes fadeIn {{
            from {{opacity: 0; transform: scale(0.95);}}
            to {{opacity: 1; transform: scale(1);}}
        }}
        </style>
        """, unsafe_allow_html=True)

        # ================= VISUALIZATION =================
        if is_ml:
            st.subheader("📊 Most Important Words (ML - TF-IDF)")

            X = vectorizer.transform([st.session_state.input_text.lower()])
            names = vectorizer.get_feature_names_out()

            if X.nnz > 0:
                df_plot = pd.DataFrame({
                    "Word": names[X.indices],
                    "Importance": X.data
                }).sort_values("Importance", ascending=False).head(20)

                chart = alt.Chart(df_plot).mark_bar().encode(
                    x="Importance",
                    y=alt.Y("Word", sort='-x')
                )

                st.altair_chart(chart, use_container_width=True)

                # Extract top keywords for Gemini
                top_keywords = ", ".join(df_plot["Word"].tolist())

                summary = generate_summary(
                    top_keywords,
                    label,
                    model_type,
                    dataset
                )

                st.subheader("🧠 AI Explanation")
                st.write(summary)

            else:
                st.info("No important words detected")

        else:
            st.subheader("☁️ Word Cloud (DL - BiLSTM Input)")

            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                max_words=20
            ).generate(st.session_state.input_text)

            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

            # Extract top words from wordcloud
            top_keywords = ", ".join(list(wc.words_.keys())[:20])
            summary = generate_summary(
                    top_keywords,
                    label,
                    model_type,
                    dataset
                )

            st.subheader("🧠 AI Explanation")
            st.write(summary)

            

# =========================================================
# GOOGLE SHEETS CONNECTION (FIXED)
# =======================fAI Explanation (Gemini)==================================
# =========================================================
# GOOGLE SHEETS CONNECTION (STREAMLIT CLOUD)
# =========================================================
import gspread
from google.oauth2.service_account import Credentials
import datetime

SHEET_ID = "1UklQD2URWeB9l6X4hTnDc9YRcNJA0d7BeOx7wXTz280"
WORKSHEET_NAME = "X_Feedback_Dashboard"

sheet = None

try:
    creds_dict = dict(st.secrets["gcp_service_account"])

    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )

    client = gspread.authorize(creds)

    spreadsheet = client.open_by_key(SHEET_ID)

    sheet = spreadsheet.worksheet(WORKSHEET_NAME)

    #st.success("✅ Google Sheet connected successfully")

except Exception as e:
    sheet = None
    st.error(f"Google Sheet connection failed: {e}")

# =========================================================
st.markdown("---")
st.header("🧠 Post-Prediction Questionnaire")
st.subheader("System Evaluation (Likert Scale)")

if "questionnaire_submitted" not in st.session_state:
    st.session_state.questionnaire_submitted = False

with st.form("feedback_form"):

    st.subheader("User Profile")

    age = st.selectbox("Age Group", ["18–24","25–34","35–44","45–54","55+"])
    gender = st.selectbox("Gender", ["Male","Female","Other","Prefer not to say"])
    education = st.selectbox(
        "Education Level",
        ["High School","Undergraduate","Postgraduate","PhD","Other"]
    )

    field = st.selectbox(
        "Field of Study",
        ["AI/CS","Social Sciences","Journalism","Business","Other"]
    )

    ai_level = st.selectbox(
        "AI Knowledge",
        ["None","Basic","Intermediate","Advanced"]
    )

    news_freq = st.selectbox(
        "News Consumption Frequency",
        ["Rarely","Weekly","Daily","Multiple times/day"]
    )

    st.subheader("System Evaluation (1–5)")

    st.markdown("""
**1 = Strongly Disagree**  
**2 = Disagree**  
**3 = Neutral**  
**4 = Agree**  
**5 = Strongly Agree**
""")

    # -------------------------------
    # SUS-Inspired Usability Questions
    # -------------------------------
    st.markdown("### Usability (SUS-Inspired)")

    q1 = st.slider(
        "The dashboard was easy to use.",
        1, 5, 3
    )

    q2 = st.slider(
        "I was able to navigate the dashboard without difficulty.",
        1, 5, 3
    )

    q3 = st.slider(
        "I learned how to use the dashboard quickly.",
        1, 5, 3
    )

    q4 = st.slider(
        "The dashboard interface was clear and well organised.",
        1, 5, 3
    )

    q5 = st.slider(
        "The system responded quickly during use.",
        1, 5, 3
    )

    # -------------------------------
    # Explainability
    # -------------------------------
    st.markdown("### Explainability")

    q6 = st.slider(
        "The model predictions were clearly presented.",
        1, 5, 3
    )

    q7 = st.slider(
        "The visualisations helped me understand the prediction results.",
        1, 5, 3
    )

    # -------------------------------
    # Trust
    # -------------------------------
    st.markdown("### Trust")

    q8 = st.slider(
        "I trust the system’s predictions.",
        1, 5, 3
    )

    # -------------------------------
    # Usefulness
    # -------------------------------
    st.markdown("### Usefulness")

    q9 = st.slider(
        "This system is useful for identifying fake news/disinformation.",
        1, 5, 3
    )

    q10 = st.slider(
        "The dashboard helped me make better decisions about whether news content was trustworthy.",
        1, 5, 3
    )

    # -------------------------------
    # Adoption
    # -------------------------------
    st.markdown("### Future Adoption")

    q11 = st.slider(
        "I would use this system again in the future.",
        1, 5, 3
    )

    comments = st.text_area(
        "What improvements would you suggest for this dashboard?"
    )

    submit = st.form_submit_button("Submit Feedback")

# =========================================================
# SAVE TO GOOGLE SHEETS
# =========================================================
if submit:

    try:
        if sheet is not None:
            sheet.append_row([
                str(datetime.datetime.now()),
                model_type,
                dataset,
                age,
                gender,
                education,
                field,
                ai_level,
                news_freq,

                # SUS-inspired usability
                q1,  # Ease_of_Use
                q2,  # Navigation
                q3,  # Learnability
                q4,  # Interface_Design
                q5,  # System_Speed

                # Explainability
                q6,  # Prediction_Clarity
                q7,  # Visualization_Helpfulness

                # Trust
                q8,  # Trust

                # Usefulness
                q9,  # Usefulness
                q10, # Decision_Support

                # Adoption
                q11, # Reuse_Intention

                comments
            ])

            st.success("✅ Feedback saved")

        else:
            st.warning("⚠ Google Sheet not connected. Data not saved.")

        st.session_state.questionnaire_submitted = True

    except Exception as e:
        st.error(f"Error saving feedback: {e}")