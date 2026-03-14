from sklearn.feature_extraction.text import TfidfVectorizer

def apply_tfidf(text_data):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,      # increased vocabulary
        ngram_range=(1,2),       # unigrams + bigrams
        min_df=2,                # keep slightly rarer useful words
        max_df=0.9,              # ignore extremely common words
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True
    )

    X = vectorizer.fit_transform(text_data)

    return X, vectorizer