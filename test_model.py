from models.data_loader import load_dataset
from models.preprocessing import clean_text
from models.tfidf_model import apply_tfidf
from models.classification import train_classifier
from sklearn.model_selection import cross_val_score

# Load dataset
df = load_dataset()

# Convert categories to uppercase
df["category"] = df["category"].str.upper()

# -----------------------------
# CATEGORY MERGING (same as app.py)
# -----------------------------
category_mapping = {

    # IT
    "PYTHON DEVELOPER": "IT",
    "JAVA DEVELOPER": "IT",
    "FRONTEND DEVELOPER": "IT",
    "BACKEND DEVELOPER": "IT",
    "FULL STACK DEVELOPER": "IT",
    "DEVOPS ENGINEER": "IT",
    "MACHINE LEARNING ENGINEER": "IT",
    "DATA SCIENTIST": "IT",
    "DATA SCIENCE": "IT",
    "CLOUD ENGINEER": "IT",
    "DATABASE": "IT",
    "HADOOP": "IT",
    "INFORMATION-TECHNOLOGY": "IT",

    # FINANCE
    "ACCOUNTANT": "FINANCE",
    "BANKING": "FINANCE",
    "FINANCE": "FINANCE",

    # EDUCATION
    "TEACHER": "EDUCATION",
    "ARTS": "EDUCATION",

    # SALES
    "SALES": "SALES",
    "BUSINESS-DEVELOPMENT": "SALES",
    "PUBLIC-RELATIONS": "SALES",

    # ENGINEERING
    "ENGINEERING": "ENGINEERING",
    "MECHANICAL ENGINEER": "ENGINEERING",
    "CIVIL ENGINEER": "ENGINEERING",
    "ELECTRICAL ENGINEERING": "ENGINEERING",

    # HEALTHCARE
    "HEALTHCARE": "HEALTHCARE",
    "FITNESS": "HEALTHCARE",

    # HR
    "HR": "HR"
}

df["category"] = df["category"].replace(category_mapping)

# Remove rare categories (same as app.py)
df = df.groupby("category").filter(lambda x: len(x) >= 40)

# -----------------------------
# TEXT CLEANING
# -----------------------------
df["clean_text"] = df["resume_text"].apply(clean_text)

# -----------------------------
# TF-IDF
# -----------------------------
X, vectorizer = apply_tfidf(df["clean_text"])

# Labels
y = df["category"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
model, accuracy, report = train_classifier(X, y)

print("Model Accuracy:", accuracy * 100)
print(report)

# -----------------------------
# CROSS VALIDATION
# -----------------------------
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv)
print(report, "\n")

print("Cross Validation Accuracy:", scores.mean() * 100)