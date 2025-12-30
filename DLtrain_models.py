# =========================================================
# DLtrain_models.py (FINAL – ISOT/EUvsISOT FIXED, map_labels included)
# =========================================================

import re, time, tracemalloc, pickle, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

nltk.download('stopwords')
nltk.download('wordnet')

# =========================================================
# 0. PARAMETERS
# =========================================================
VOCAB_SIZE = 1000
MAX_LEN = 700
EMBEDDING_DIM = 16
LSTM_UNITS = 128
DROPOUT = 0.3
EPOCHS = 30

MODEL_DIR = "dl_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# 1. GLOBAL LABEL SEMANTICS
# =========================================================
LABEL_MAP = {
    0: "True",
    1: "Fake / Disinformation"
}

# =========================================================
# 2. TEXT CLEANING
# =========================================================
def clean_text_column(df, text_col):
    df = df.copy()
    df["text"] = (
        df[text_col]
        .astype(str)
        .apply(lambda x: re.sub(r"[^\x00-\x7F]+", " ", x))
        .str.strip()
    )
    return df

# =========================================================
# 3. MAP LABELS
# =========================================================
def map_labels(df, label_col, label_map):
    """
    Safely map labels in a DataFrame to integers.
    Drops rows that cannot be mapped.
    """
    df = df.copy()
    df['label'] = df[label_col].map(label_map)
    df = df[df['label'].notna()]  # drop unmapped rows
    df['label'] = df['label'].astype(int)
    df.reset_index(drop=True, inplace=True)
    return df

# =========================================================
# 4. LABEL FIX FOR ISOT AND EUvsISOT
# =========================================================
def standardize_labels(df, dataset_name):
    df = df.copy()
    if dataset_name == "ISOT":
        # Already assigned: True=0, Fake=1
        pass
    elif dataset_name == "EUvsISOT":
        df['label'] = df['class'].map({'True':0, 'disinformation':1})
    else:
        # For other datasets, ensure 0=True, 1=Fake/Disinformation
        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)
        elif 'class' in df.columns:
            df['label'] = df['class'].map({'true':0, 'support':0, 'disinformation':1})
    df = df[df['label'].notna()]
    df['label'] = df['label'].astype(int)
    df.reset_index(drop=True, inplace=True)
    return df

# =========================================================
# 5. BiLSTM TRAINING FUNCTION
# =========================================================
def run_bilstm(df, name):
    print("\n==============================")
    print(f"BiLSTM Training on {name}")
    print("==============================")

    # Fix labels for ISOT/EUvsISOT
    df = standardize_labels(df, name)

    texts = df["text"].values
    labels = df["label"].astype(int).values

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    x_train, x_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=434, stratify=labels, shuffle=True
    )

    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(LSTM_UNITS)),
        Dropout(DROPOUT),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    tracemalloc.start()
    start_time = time.time()

    model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test),
              callbacks=[early_stop], verbose=2)

    end_time = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Evaluation
    y_prob = model.predict(x_test, verbose=0)
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Time (s) : {end_time - start_time:.2f}")
    print(f"Memory(MB): {peak / 1024 / 1024:.2f}")

    print("\nClassification Report (0=True, 1=Fake):\n")
    print(classification_report(y_test, y_pred, target_names=["True", "Fake / Disinformation"], digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix – {name}")
    plt.xticks([0, 1], ["True", "Fake"])
    plt.yticks([0, 1], ["True", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Save model & tokenizer
    model.save(os.path.join(MODEL_DIR, f"{name}_bilstm.h5"))
    with open(os.path.join(MODEL_DIR, f"{name}_tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    return {
        "Dataset": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Time(s)": end_time - start_time,
        "Memory(MB)": peak / 1024 / 1024
    }

# =========================================================
# 6. LOAD DATASETS (local CSVs)
# =========================================================

# FA-KES
fa_kes = pd.read_csv("FA-KES-Dataset.csv", encoding='ISO-8859-1')
fa_kes = map_labels(fa_kes, 'labels', {'Fake':1, 'TRUE':0})
fa_kes = clean_text_column(fa_kes, 'text')

# Pandy
pandy = pd.read_csv("Pandy-Dataset.csv", encoding='ISO-8859-1')
pandy = map_labels(pandy, 'label', {'FAKE':1, 'REAL':0})
pandy = clean_text_column(pandy, 'text')

# EUvsIPF
eu_ipf = pd.read_csv("EUvsIPF-Dataset.csv", encoding='latin1')
eu_ipf = map_labels(eu_ipf, 'class', {'disinformation':1, 'true':0})
eu_ipf = clean_text_column(eu_ipf, 'text_english')

# EUvsDisinfo
eu_disinfo = pd.read_csv("EUvsDisinfo-Dataset.csv", encoding='latin1')
eu_disinfo = map_labels(eu_disinfo, 'class', {'disinformation':1, 'support':0})
eu_disinfo = clean_text_column(eu_disinfo, 'text_english')

# ISOT
true_df = pd.read_csv("ISOT_True.csv")
fake_df = pd.read_csv("ISOT_Fake.csv")
true_df['label'] = 0
fake_df['label'] = 1
isot = pd.concat([true_df, fake_df], ignore_index=True)
isot = clean_text_column(isot, 'text')

# EUvsISOT
eu_isot = pd.read_csv("EUvsISOT-Dataset.csv", encoding='latin1')
eu_isot = map_labels(eu_isot, 'class', {'disinformation':1, 'True':0})
eu_isot = clean_text_column(eu_isot, 'text_english')

datasets = {
    'FA-KES': fa_kes,
    'Pandy': pandy,
    'EUvsIPF': eu_ipf,
    'EUvsDisinfo': eu_disinfo,
    'ISOT': isot,
    'EUvsISOT': eu_isot
}

# =========================================================
# 7. RUN ALL DATASETS
# =========================================================
if __name__ == "__main__":
    results = []

    for name, df in datasets.items():
        results.append(run_bilstm(df, name))

    results_df = pd.DataFrame(results)
    print("\n===== FINAL RESULTS =====")
    print(results_df)

    results_df.to_csv(os.path.join(MODEL_DIR, "bilstm_results.csv"), index=False)
