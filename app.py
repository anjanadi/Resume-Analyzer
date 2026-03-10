from reportlab.lib import styles
import streamlit as st
import pandas as pd
import io
import re

from utils.file_handler import extract_text_from_pdf
from utils.similarity_checker import calculate_similarity_score
from utils.metadata_extractor import extract_email_phone_location

from models.data_loader import load_dataset
from models.preprocessing import clean_text
from models.tfidf_model import apply_tfidf
from models.classification import train_classifier
from models.skill_alignment import generate_skill_templates

from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="AI Resume Screening System", layout="wide")

st.title("🤖 AI Resume Screening & Skill Gap Analyzer")

st.markdown("""
This system uses **Machine Learning + Skill Intelligence**
to evaluate resumes for job positions.
""")


# -----------------------------
# EXPERIENCE EXTRACTOR
# -----------------------------
def extract_experience(resume_text):

    text = resume_text.lower()
    experience = 0

    match1 = re.search(r'(\d+(\.\d+)?)\s*(year|years)', text)
    match2 = re.search(r'(\d+)\+\s*(year|years)', text)

    if match1:
        experience = float(match1.group(1))

    elif match2:
        experience = float(match2.group(1))

    return experience


# -----------------------------
# PDF GENERATOR
# -----------------------------
def create_pdf(df):

    buffer = io.BytesIO()

    styles = getSampleStyleSheet()

    title = Paragraph("AI Resume Screening Results", styles["Title"])

    # convert dataframe rows to paragraphs for wrapping
    data = [df.columns.tolist()]
    for row in df.values:
        data.append([Paragraph(str(cell), styles["Normal"]) for cell in row])

    # control column widths
    col_widths = [80, 80, 60, 70, 90, 90, 90, 120, 80]

    table = Table(data, colWidths=col_widths, repeatRows=1)

    table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.grey),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),

            ("ALIGN", (0,0), (-1,-1), "CENTER"),

            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),

            ("BOTTOMPADDING", (0,0), (-1,0), 10),

            ("BACKGROUND", (0,1), (-1,-1), colors.beige),

            ("GRID", (0,0), (-1,-1), 1, colors.black),
        ]))

    pdf = SimpleDocTemplate(buffer, pagesize=letter)

    pdf.build([title, table])

    buffer.seek(0)

    return buffer


# -----------------------------
# LOAD AI MODEL
# -----------------------------
@st.cache_resource
def load_ai_system():

    df = load_dataset()

    df["category"] = df["category"].str.upper()

    # -----------------------------
    # CATEGORY MERGING
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

    # -----------------------------
    # REMOVE RARE CATEGORIES
    # -----------------------------
    df = df.groupby("category").filter(lambda x: len(x) >= 15)

    # -----------------------------
    # CLEAN TEXT
    # -----------------------------
    df["cleaned_resume"] = df["resume_text"].apply(clean_text)

    # -----------------------------
    # TF-IDF
    # -----------------------------
    X, vectorizer = apply_tfidf(df["cleaned_resume"])

    y = df["category"]

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------
    model, accuracy, report = train_classifier(X, y)

    templates = generate_skill_templates(df)

    return model, vectorizer, templates, accuracy


model, vectorizer, templates, model_accuracy = load_ai_system()

st.sidebar.markdown("### Model Information")
st.sidebar.write("Classification Accuracy:", round(model_accuracy * 100, 2), "%")


# -----------------------------
# INPUT SECTION
# -----------------------------
st.header("📄 Job Description")

jd_input = st.text_area("Paste Job Description", height=200)

resume_files = st.file_uploader(
    "Upload Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)

required_experience = st.number_input(
    "Required Experience (Years)",
    min_value=0.0,
    step=0.5
)


# -----------------------------
# PROCESS RESUMES
# -----------------------------
if jd_input and resume_files:

    results = []

    for file in resume_files:

        resume_text = extract_text_from_pdf(file)

        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(jd_input)

        input_vector = vectorizer.transform([resume_clean])

        predicted_role = model.predict(input_vector)[0]

        confidence_scores = model.decision_function(input_vector)

        raw_confidence = abs(confidence_scores.max())

        prediction_confidence = (raw_confidence / (raw_confidence + 1)) * 100


        # -----------------------------
        # SKILL ALIGNMENT
        # -----------------------------
        if predicted_role in templates:

            required_skills = templates[predicted_role]

            matched_skills = [
                skill for skill in required_skills
                if skill in resume_text.lower()
            ]

            missing_skills = list(set(required_skills) - set(matched_skills))

            skill_score = (len(matched_skills) / len(required_skills)) * 100

        else:

            required_skills = []
            matched_skills = []
            missing_skills = []
            skill_score = 0


        # -----------------------------
        # JD SIMILARITY
        # -----------------------------
        tfidf_score = calculate_similarity_score(jd_clean, resume_clean)


        # -----------------------------
        # EXPERIENCE MATCH
        # -----------------------------
        candidate_experience = extract_experience(resume_text)

        if required_experience == 0:
            experience_score = 100
        else:
            experience_score = min((candidate_experience / required_experience) * 100, 100)


        # -----------------------------
        # FINAL SCORE
        # -----------------------------
        final_score = round(
            (0.5 * skill_score) +
            (0.25 * tfidf_score) +
            (0.15 * prediction_confidence) +
            (0.10 * experience_score),
            2
        )


        email, phone, location = extract_email_phone_location(resume_text)

        status = "Selected ✅" if final_score >= 40 else "Rejected ❌"

        suggestion = (
            "Improve skills: " + ", ".join(missing_skills)
            if missing_skills else "Strong skill alignment"
        )


        results.append({

        "Resume": file.name,
        "Predicted Role": predicted_role,
        "Prediction Confidence (%)": round(prediction_confidence, 2),
        "Required Skills": ", ".join(required_skills),
        "Matched Skills": ", ".join(matched_skills),
        "Missing Skills": ", ".join(missing_skills),
        "Skill Alignment Score (%)": round(skill_score, 2),
        "JD Similarity Score (%)": tfidf_score,
        "Candidate Experience (Years)": candidate_experience,
        "Final Score (%)": final_score,
        "Decision": status,
        "Suggestion": suggestion,
        "Email": email,
        "Phone": phone,
        "Location": location
    })


    df_results = pd.DataFrame(results)

    st.header("📊 Resume Evaluation Results")

# Sort candidates by score
    df_results = df_results.sort_values(by="Final Score (%)", ascending=False)

# Arrange columns in proper order
    df_results = df_results[[
        "Resume",
        "Predicted Role",
        "Final Score (%)",
        "Decision",
        "Candidate Experience (Years)",
        "Skill Alignment Score (%)",
        "JD Similarity Score (%)",
        "Prediction Confidence (%)",
        "Matched Skills",
        "Missing Skills",
        "Suggestion",
        "Email",
        "Phone",
        "Location"
    ]]

# Display table
    st.dataframe(
    df_results,
    use_container_width=True
)

    # -----------------------------
    # DOWNLOAD EXCEL
    # -----------------------------
    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False)

    excel_buffer.seek(0)

    st.download_button(
        label="📥 Download Results as Excel",
        data=excel_buffer,
        file_name="resume_screening_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


    # -----------------------------
    # DOWNLOAD PDF
    # -----------------------------
    pdf_df = df_results[[
    "Resume",
    "Predicted Role",
    "Final Score (%)",
    "Decision",
    "Candidate Experience (Years)",
    "Skill Alignment Score (%)",
    "JD Similarity Score (%)",
    "Email",
    "Phone"
]]

    pdf_file = create_pdf(pdf_df)
    st.download_button(
        label="📄 Download Results as PDF",
        data=pdf_file,
        file_name="resume_screening_results.pdf",
        mime="application/pdf"
    )

    # title = Paragraph("AI Resume Screening Results Report", styles['Title'])

elif jd_input and not resume_files:

    st.info("Please upload at least one resume.")

elif resume_files and not jd_input:

    st.info("Please paste the job description.")