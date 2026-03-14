
AI Resume Screening & Skill Gap Analyzer
📌 Overview

The AI Resume Screening & Skill Gap Analyzer is a machine learning–based system that automates the resume evaluation process for recruitment. It analyzes candidate resumes against a job description and generates insights such as role prediction, skill alignment, missing skills, experience matching, and final candidate scoring.

The system helps recruiters quickly filter candidates by identifying relevant skills and ranking resumes based on multiple evaluation parameters.

🎯 Objectives
The main objectives of this project are:
Automate resume screening using machine learning.
Classify resumes into appropriate job roles.
Compare candidate resumes with job descriptions.
Identify skill gaps and missing competencies.
Provide a final evaluation score for recruitment decisions.
Extract candidate contact details automatically.

🚀 Features
Resume Classification
Uses a supervised machine learning model to classify resumes into predefined job categories.
Skill Gap Analysis
The system identifies:
Matched skills
Missing skills
Skill alignment score
Job Description Similarity
Calculates similarity between the job description and candidate resume using TF-IDF vectorization and cosine similarity.
Experience Detection
Automatically extracts years of experience mentioned in the resume.
Contact Information Extraction
Extracts important candidate details such as:
Email address
Phone number
Location
Hybrid Candidate Scoring
Each resume is evaluated using multiple parameters:
Skill alignment
Job description similarity
Prediction confidence
Experience match

A final score is generated to help recruiters make faster decisions.

| Component              | Technique Used                |
| ---------------------- | ----------------------------- |
| Text Processing        | NLP preprocessing             |
| Feature Extraction     | TF-IDF                        |
| Classification Model   | Linear Support Vector Machine |
| Similarity Measurement | Cosine Similarity             |


Technologies Used
Programming Language
Python
Libraries
Scikit-learn
Pandas
NumPy
NLTK
PyPDF2
Streamlit
ReportLab

Tools
VS Code
Git
GitHub

📊 Model Performance
The resume classification model achieved approximately:
~73% – 80% accuracy

📂 Project Structure
Resume-Analyzer
│
├── app.py
├── test_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── tfidf_model.py
│   ├── classification.py
│   └── skill_alignment.py
│
├── utils
│   ├── file_handler.py
│   ├── similarity_checker.py
│   └── metadata_extractor.py
│
└── data
    └── Resume.csv


   ⚙️ Installation
Clone the repository:-

git clone https://github.com/sub2004-nanda/Resume-Analyzing-System

Navigate to project folder:-

cd Resume-Analyzer

Create virtual environment:-

python -m venv env:-

Activate environment in Windows:-

env\Scripts\activate

Install dependencies:-

pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py







