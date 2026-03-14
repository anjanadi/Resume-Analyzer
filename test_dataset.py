from models.data_loader import load_dataset
from models.skill_alignment import (
    generate_skill_templates,
    extract_skills_from_text,
    calculate_alignment
)

import pandas as pd

# Load dataset
df = load_dataset()

# 🔹 Check dataset size after Step 3
print("Dataset size:", len(df))
print("Total categories:", df["category"].nunique())

# Generate automatic templates
templates = generate_skill_templates(df, top_n=5)

alignment_results = []

for index, row in df.iterrows():

    category = str(row["category"]).upper()
    resume_text = str(row["resume_text"])

    # Skip categories without templates
    if category not in templates:
        continue

    required_skills = templates[category]

    matched = extract_skills_from_text(resume_text, required_skills)

    score = calculate_alignment(matched, required_skills)

    alignment_results.append({
        "category": category,
        "alignment_score": score
    })


alignment_df = pd.DataFrame(alignment_results)

print("\nAverage Alignment (Auto Templates):\n")

print(
    alignment_df
    .groupby("category")["alignment_score"]
    .mean()
    .sort_values(ascending=False)
)