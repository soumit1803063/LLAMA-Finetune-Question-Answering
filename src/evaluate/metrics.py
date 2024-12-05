from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(real_answer, generated_answer):
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the answers into TF-IDF vectors
    vectors = vectorizer.fit_transform([real_answer, generated_answer])

    # Compute cosine similarity
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return similarity
