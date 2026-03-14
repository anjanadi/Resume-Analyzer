import pandas as pd
from models.preprocessing import clean_text
from datasets import load_dataset as hf_load_dataset


# -------------------------
# Load your existing dataset
# -------------------------
def load_dataset():

    rows = []

    with open("data/Resume.csv", "r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]

            parts = line.split(",", 4)

            if len(parts) == 5:
                rows.append(parts)

    df = pd.DataFrame(rows, columns=[
        "resume_id",
        "category",
        "resume_text",
        "skills_list",
        "experience_years"
    ])

    # Remove header row
    df = df[df["resume_id"] != "resume_id"]

    # Convert numeric columns
    df["resume_id"] = pd.to_numeric(df["resume_id"], errors="coerce")
    df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")

    df = df.dropna(subset=["resume_id"])

    # Remove rare categories
    df = df.groupby("category").filter(lambda x: len(x) >= 40)  

    # Text preprocessing
    df["clean_text"] = df["resume_text"].apply(clean_text)

    # -------------------------
    # Merge similar categories
    # -------------------------
    category_map = {
        "INFORMATION TECHNOLOGY": "IT",
        "INFORMATION-TECHNOLOGY": "IT",
    "HUMAN RESOURCES": "HR",
    "DIGITAL-MEDIA": "DIGITAL MEDIA",
    "DESIGNER": "DESIGNING",
    "CONSTRUCTION": "BUILDING AND CONSTRUCTION",
    "PUBLIC RELATIONS": "PR"
    }

    df["category"] = df["category"].replace(category_map)

    # Keep required columns
    df = df[["category", "resume_text", "clean_text", "skills_list", "experience_years"]]

    # -------------------------
    # Load ResumeAtlas dataset
    # -------------------------
    atlas = hf_load_dataset("ahmedheakl/resume-atlas")

    atlas_df = pd.DataFrame(atlas["train"])

    print("Atlas columns:", atlas_df.columns)

    atlas_df = atlas_df.rename(columns={
        "Text": "resume_text",
        "Category": "category"
    })

    atlas_df["clean_text"] = atlas_df["resume_text"].apply(clean_text)

    atlas_df["skills_list"] = ""
    atlas_df["experience_years"] = 0

    atlas_df = atlas_df[["category", "resume_text", "clean_text", "skills_list", "experience_years"]]

    # -------------------------
    # Merge datasets
    # -------------------------
    df = pd.concat([df, atlas_df], ignore_index=True)

    # Remove duplicate resumes
    df = df.drop_duplicates(subset=["clean_text"])
    
    return df