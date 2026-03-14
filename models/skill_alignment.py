import re

# Technical skill dictionary
TECH_SKILLS = {

# Programming
"python","java","c","c++","c#","golang","rust","kotlin","swift",
"javascript","typescript","php","ruby","scala","matlab",

# Web Development
"html","css","sass","bootstrap","tailwind",
"react","angular","vue","nextjs","node","express",
"flask","django","spring","laravel",

# Databases
"sql","mysql","postgresql","mongodb","redis","sqlite",
"oracle","cassandra","dynamodb",

# Data Science
"pandas","numpy","scikit-learn","matplotlib","seaborn",
"machine learning","deep learning","nlp","computer vision",
"data analysis","data visualization",

# AI / ML frameworks
"tensorflow","pytorch","keras","xgboost","lightgbm",

# Cloud
"aws","azure","gcp","google cloud","cloud computing",

# DevOps
"docker","kubernetes","jenkins","terraform","ansible",
"ci cd","devops","linux","bash","shell scripting",

# Big Data
"hadoop","spark","hive","kafka","airflow",

# Security
"network security","penetration testing","ethical hacking",
"cybersecurity","cryptography",

# Mobile
"android","ios","flutter","react native",

# Testing
"unit testing","selenium","cypress","pytest","automation testing",

# APIs
"rest","rest api","graphql","microservices",

# Version control
"git","github","gitlab","bitbucket"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# Extract skills from resume
def extract_skills(text):

    text = clean_text(text)

    found_skills = []

    for skill in TECH_SKILLS:
        if skill in text:
            found_skills.append(skill)

    return found_skills


# Generate skill templates per category
def generate_skill_templates(df, top_n=5):

    templates = {}

    for category in df["category"].unique():

        resumes = df[df["category"] == category]

        skill_counts = {}

        for text in resumes["resume_text"]:
            skills = extract_skills(text)

            for skill in skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1

        # Sort by frequency
        sorted_skills = sorted(skill_counts, key=skill_counts.get, reverse=True)

        templates[category.upper()] = sorted_skills[:top_n]

    return templates


# Extract required skills from text
def extract_skills_from_text(text, required_skills):

    text = text.lower()

    matched = []

    for skill in required_skills:
        if skill.lower() in text:
            matched.append(skill)

    return matched


# Calculate alignment score
def calculate_alignment(matched_skills, required_skills):

    if len(required_skills) == 0:
        return 0

    return len(matched_skills) / len(required_skills)