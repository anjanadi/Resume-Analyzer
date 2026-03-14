from datasets import load_dataset
import pandas as pd
from models.preprocessing import clean_text


def load_atlas_dataset():

    dataset = load_dataset("ahmedheakl/resume-atlas")

    df = pd.DataFrame(dataset["train"])

    df = df.rename(columns={
        "text": "resume_text",
        "label": "category"
    })

    df["resume_text"] = df["resume_text"].apply(clean_text)

    return df