import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = str(text).lower()

    # remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # remove urls
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # remove phone numbers
    text = re.sub(r'\d{10,}', ' ', text)

    # remove non alphabet characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    # remove stopwords
    words = [w for w in words if w not in stop_words]

    return " ".join(words)