import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load your training dataset text
df = pd.read_csv("Pandy-Dataset_sample20.csv")  # Or full dataset
texts = df['text'].astype(str).tolist()

# Create and fit vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
vectorizer.fit(texts)

# Save the fitted vectorizer
with open("pandy_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("vedctorizer fitted and saved as pandy_vectorizer.pkl")
import pickle

with open("pandy_vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

print(hasattr(vec, "idf_"))