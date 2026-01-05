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
