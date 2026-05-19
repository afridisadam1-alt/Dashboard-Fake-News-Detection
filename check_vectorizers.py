from pathlib import Path
import pickle

vector_file = "pandy_vectorizer.pkl"
vector_path = Path(__file__).parent / vector_file

print("Vectorizer exists?", vector_path.exists())

with open(vector_path, "rb") as f:
    vectorizer = pickle.load(f)

print("Vectorizer type:", type(vectorizer))
print("Vectorizer fitted?", hasattr(vectorizer, "idf_"))
