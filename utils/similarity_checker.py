from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity_score(jd_text, resume_text):
    try:
        documents = [jd_text, resume_text]

        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),   # BIG improvement
            max_features=5000    # reduces noise
        )

        tfidf_matrix = vectorizer.fit_transform(documents)
        raw_score = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        # 🔥 NORMALIZATION (Fix-1)
        boosted_score = raw_score * 100 * 1.5
        final_score = min(round(boosted_score, 2), 100)

        return final_score

    except Exception as e:
        print(f"Similarity error: {e}")
        return 0.0
