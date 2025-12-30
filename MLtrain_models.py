# =============================
# train_models.py
# =============================
import re, pickle, time, tracemalloc, os
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.stem import WordNetLemmatizer
import gdown

nltk.download('stopwords')
nltk.download('wordnet')

# =============================
# 0. Setup paths & functions
# =============================
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

def map_labels(df, label_col, label_map):
    """Map column to integer labels safely."""
    df['label'] = df[label_col].map(label_map)
    df = df[df['label'].notna()]  # drop rows that couldn't be mapped
    df['label'] = df['label'].astype(int)
    df.reset_index(drop=True, inplace=True)  # reset index to avoid gaps
    return df

def clean_text_column(df, text_col='text'):
    df['text'] = df[text_col].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', str(x)))
    return df

def download_csv_from_drive(file_id, output_file):
    if not os.path.exists(output_file):
        print(f"Downloading {output_file} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)
    else:
        print(f"{output_file} already exists. Skipping download.")
    return output_file

# =============================
# 1. Load datasets from Google Drive
# =============================
file_ids = {
    "FA-KES": "1GMfyApxEy35Cx9W1bl0STkE0cv7uuzzj",
    "Pandy": "119FKUobBhiSxphIorkpJRD0crCnj1GPA",
    "EUvsIPF": "1-xUFXo7FfNenF9F3iHY_5z8EM29vpXl_",
    "EUvsDisinfo": "1w4KLbjxgE6_0-dgD8ttmPXUA096FCXI6",
    "ISOT_True": "1TQJFN024YBFqhLO8TFD7LL3iTbvH1NXH",
    "ISOT_Fake": "1wnyPZr9-cy0TXwDVjRa2vE1AM9-1hPie",
    "EUvsISOT": "1YzfBjqItzsrRC9aYmGz4WpEnXTAkDsx6"
}

# Download & load FA-KES
fa_kes_file = download_csv_from_drive(file_ids["FA-KES"], "FA-KES-Dataset.csv")
fa_kes = pd.read_csv(fa_kes_file, encoding='ISO-8859-1')
fa_kes = map_labels(fa_kes, 'labels', {'Fake':1, 'TRUE':0})
fa_kes = clean_text_column(fa_kes, 'text')

# Download & load Pandy
pandy_file = download_csv_from_drive(file_ids["Pandy"], "Pandy-Dataset.csv")
pandy = pd.read_csv(pandy_file, encoding='ISO-8859-1')
pandy = map_labels(pandy, 'label', {'FAKE':1, 'REAL':0})
pandy = clean_text_column(pandy, 'text')

# Download & load EUvsIPF
eu_ipf_file = download_csv_from_drive(file_ids["EUvsIPF"], "EUvsIPF-Dataset.csv")
eu_ipf = pd.read_csv(eu_ipf_file, encoding='latin1')
eu_ipf = map_labels(eu_ipf, 'class', {'disinformation':1, 'true':0})
eu_ipf = clean_text_column(eu_ipf, 'text_english')

# Download & load EUvsDisinfo
eu_disinfo_file = download_csv_from_drive(file_ids["EUvsDisinfo"], "EUvsDisinfo-Dataset.csv")
eu_disinfo = pd.read_csv(eu_disinfo_file, encoding='latin1')
eu_disinfo = map_labels(eu_disinfo, 'class', {'disinformation':1, 'support':0})
eu_disinfo = clean_text_column(eu_disinfo, 'text_english')

# Download & load ISOT True & Fake
isot_true_file = download_csv_from_drive(file_ids["ISOT_True"], "ISOT_True.csv")
isot_fake_file = download_csv_from_drive(file_ids["ISOT_Fake"], "ISOT_Fake.csv")
true_df = pd.read_csv(isot_true_file)
fake_df = pd.read_csv(isot_fake_file)
true_df['label'] = 0
fake_df['label'] = 1
isot = pd.concat([true_df, fake_df], ignore_index=True)
isot = clean_text_column(isot, 'text')

# Download & load EUvsISOT
eu_isot_file = download_csv_from_drive(file_ids["EUvsISOT"], "EUvsISOT-Dataset.csv")
eu_isot = pd.read_csv(eu_isot_file, encoding='latin1')
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

# =============================
# 2. TIFN Class
# =============================
class TIFN():
    def __init__(self, data):
        self.data = data

    def Pre_Process(self):
        documents = []
        X,y=self.Collect_data()
        stemmer = WordNetLemmatizer()
        for sen in range(len(X)):
            document = re.sub(r'\W', ' ', str(X[sen]))
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
            document = re.sub(r'\s+', ' ', document, flags=re.I)
            document = re.sub(r'^b\s+', '', document)
            document = document.lower()
            document = document.split()
            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
            documents.append(document)
        return documents

    def Collect_data(self):
      global data
      data = data.copy()  # safe copy
      X = data["text"].reset_index(drop=True)
      y = data["label"].reset_index(drop=True)
      return X, y

    def Train_Save_PAC(self, model_path, vectorizer_path):
        X, y = self.Collect_data()
        X = self.Pre_Process()

        vectorizer = TfidfVectorizer(
            encoding='utf-8',
            max_df=0.9,
            min_df=5,
            ngram_range=(1,2),
            token_pattern='(?u)\\b\\w\\w+\\b',
            stop_words='english'
        )
        X_tfidf = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.25, random_state=42
        )

        clf = PassiveAggressiveClassifier(max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1scor = f1_score(y_test, y_pred)
        cnf = confusion_matrix(y_test, y_pred)

        print(f"\n=== PAC trained for {model_path.split('/')[-1]} ===")
        print("Accuracy:", acc, "Precision:", prec, "Recall:", rec, "F1:", f1scor)
        print("Confusion Matrix:\n", cnf)

        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

# =============================
# 3. Train PAC for all datasets
# =============================
for name, df in datasets.items():
    print(f"\n================ Training PAC on {name} =================")
    model_file = os.path.join(model_dir, f'{name.lower()}_pac.pkl')
    vectorizer_file = os.path.join(model_dir, f'{name.lower()}_vectorizer.pkl')
    data = df.copy()
    model = TIFN(data)
    model.Train_Save_PAC(model_file, vectorizer_file)
