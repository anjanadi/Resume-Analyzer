import streamlit as st
import pandas as pd
import io
import re

from utils.file_handler import extract_text_from_pdf
from utils.similarity_checker import calculate_similarity_score
from utils.metadata_extractor import extract_email_phone_location


# =========================
# SKILLS DATABASE
# =========================
SKILLS_DB = [
    "python", "java", "javascript", "react", "node",
    "flask", "django", "sql", "mysql", "postgresql",
    "mongodb", "html", "css", "rest api",
    "machine learning", "deep learning", "nlp",
    "docker", "aws", "git", "github"
]


# =========================
# TEXT CLEANING (FIX-2)
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)          # emails
    text = re.sub(r'\b\d{10}\b', ' ', text)      # phone numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)     # special characters
    text = re.sub(r'\s+', ' ', text)             # extra spaces

    noise_words = [
        "education", "experience", "skills",
        "projects", "certification", "declaration", "address"
    ]
    for word in noise_words:
        text = text.replace(word, '')

    return text.strip()


# =========================
# SKILL EXTRACTION & GAP
# =========================
def extract_skills(text):
    text = text.lower()
    return {skill for skill in SKILLS_DB if skill in text}


def skill_gap_analysis(job_text, resume_text):
    job_skills = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)

    matched_skills = job_skills & resume_skills
    missing_skills = job_skills - resume_skills

    return matched_skills, missing_skills


# =========================
# FIX-3: SKILL SCORE
# =========================
def calculate_skill_score(job_text, resume_text):
    job_skills = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)

    if not job_skills:
        return 0.0

    matched = job_skills & resume_skills
    skill_score = (len(matched) / len(job_skills)) * 100

    return round(skill_score, 2)


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Resume Screening Bot", layout="centered")

st.title("📑 Resume Screening AI Bot")
st.markdown("### Job Description")

jd_input = st.text_area("📝 Job Description (Max 1000 words)", height=250)
word_count = len(jd_input.split())

if word_count > 1000:
    st.warning(f"❗ Word limit exceeded ({word_count}/1000). Please reduce your job description.")
    jd_input = ""

resume_files = st.file_uploader(
    "📎 Upload Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)


# =========================
# MAIN LOGIC
# =========================
if jd_input and resume_files:
    st.markdown("## 📊 Match Scores")

    results = []

    for file in resume_files:
        resume_text = extract_text_from_pdf(file)

        # Clean text before similarity
        jd_clean = clean_text(jd_input)
        resume_clean = clean_text(resume_text)

        # TF-IDF score (Fix-1 inside function)
        tfidf_score = calculate_similarity_score(jd_clean, resume_clean)

        # Skill analysis
        matched_skills, missing_skills = skill_gap_analysis(jd_input, resume_text)
        skill_score = calculate_skill_score(jd_input, resume_text)

        # FIX-3: HYBRID FINAL SCORE
        final_score = round((0.6 * tfidf_score) + (0.4 * skill_score), 2)

        # Contact details
        email, phone, location = extract_email_phone_location(resume_text)

        # Selection logic
        status = "Selected ✅" if final_score >= 40 else "Rejected ❌"

        suggestion = (
            "Improve skills: " + ", ".join(missing_skills)
            if missing_skills else "All required skills matched"
        )

        results.append({
            "Resume": file.name,
            "TF-IDF Score": f"{tfidf_score}%",
            "Skill Match Score": f"{skill_score}%",
            "Final Score": f"{final_score}%",
            "Status": status,
            "Matched Skills": ", ".join(matched_skills),
            "Skill Gap": ", ".join(missing_skills),
            "Improvement Suggestion": suggestion,
            "Email": email,
            "Contact Number": phone,
            "Location": location
        })

    df_results = pd.DataFrame(results)
    st.table(df_results)

    # Excel download
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='Results')
    output.seek(0)

    st.download_button(
        label="📥 Download Results as Excel",
        data=output,
        file_name="resume_scores.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


elif jd_input and not resume_files:
    st.info("Upload at least one resume PDF.")

elif resume_files and not jd_input:
    st.info("Paste the job description above.")
