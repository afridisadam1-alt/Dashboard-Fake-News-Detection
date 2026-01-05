import pickle
from sklearn.utils.validation import check_is_fitted

# List of your ML vectorizer files
files = ["pandy_vectorizer.pkl", "euvsipf_vectorizer.pkl", "euvsdisinfo_vectorizer.pkl"]

for f in files:
    with open(f, "rb") as file:
        vec = pickle.load(file)
    try:
        check_is_fitted(vec)
        print(f"{f} is fitted ✅")
    except Exception as e:
        print(f"{f} is NOT fitted ❌: {e}")
GDRIVE_ML_MODELS = {
    "George McIntire": {"model": "pandy_vectorizer.pkl", "vectorizer": "pandy_vectorizer.pkl"},
    "EUvsIPF": {"model": "euvsipf_pac.pkl", "vectorizer": "euvsipf_vectorizer.pkl"},
    "EUvsDisinfo": {"model": "euvsdisinfo_pac.pkl", "vectorizer": "euvsdisinfo_vectorizer.pkl"},
}