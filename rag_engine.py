from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------
# Why these imports?
# TfidfVectorizer converts text into numerical vectors (our "embeddings")
# cosine_similarity measures how similar two vectors are (our "search")
# numpy handles the math behind the scenes
# ----------------------

def load_documents(filepath: str) -> list[dict]:
    """
    Reads the knowledge base file and splits it into individual documents.
    Each document becomes a dict with a title and content.
    """
    with open(filepath, "r") as f:
        raw = f.read()

    # Split on "DOCUMENT:" to separate each entry
    # The [1:] skips the empty string before the first "DOCUMENT:"
    raw_docs = raw.strip().split("DOCUMENT:")[1:]

    documents = []
    for doc in raw_docs:
        lines = doc.strip().split("\n")
        # First line is the title (e.g. "UAE Visa Policy")
        title = lines[0].strip()
        # Everything after is the content, joined back into a paragraph
        content = " ".join(lines[1:]).strip()
        documents.append({"title": title, "content": content})

    return documents
    # Result: a list of dicts like [{"title": "UAE Visa Policy", "content": "..."}, ...]


def build_index(documents: list[dict]):
    """
    Converts all document contents into TF-IDF vectors.
    This is your "index" — the searchable version of your knowledge base.
    Think of it like building a library catalog.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    # stop_words="english" removes words like "the", "is", "and"
    # These words appear everywhere and carry no search meaning

    contents = [doc["content"] for doc in documents]
    # Extract just the text content from each document dict

    matrix = vectorizer.fit_transform(contents)
    # fit_transform does two things:
    # "fit" → learns the vocabulary from all documents
    # "transform" → converts each document into a vector of numbers
    # Result: a matrix where each row = one document, each column = one word's score

    return vectorizer, matrix
    # We return both because we need the vectorizer later to convert queries too


def search(query: str, documents: list[dict], vectorizer, matrix, top_k: int = 2) -> list[dict]:
    """
    Given a question, find the most relevant documents.
    This is the core of RAG — retrieval.
    """
    query_vector = vectorizer.transform([query])
    # Convert the query into the same vector format as our documents
    # We use transform (not fit_transform) because the vocabulary is already learned

    similarities = cosine_similarity(query_vector, matrix)[0]
    # Cosine similarity measures the angle between two vectors
    # Score of 1.0 = identical meaning, 0.0 = completely unrelated
    # [0] because cosine_similarity returns a 2D array, we want the first row

    top_indices = np.argsort(similarities)[::-1][:top_k]
    # argsort returns indices that would sort the array (lowest to highest)
    # [::-1] reverses it (highest to lowest)
    # [:top_k] takes only the top K results

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            # Only include results with some actual similarity (score > 0)
            # A score of 0 means zero overlap — not useful context
            results.append({
                "title": documents[idx]["title"],
                "content": documents[idx]["content"],
                "score": round(float(similarities[idx]), 3)
                # round() for clean display, float() because numpy numbers
                # don't serialize to JSON cleanly
            })

    return results


def format_context(results: list[dict]) -> str:
    """
    Converts retrieved documents into a text block for the LLM prompt.
    This is what gets injected into the agent's thinking prompt.
    """
    if not results:
        return "No relevant information found in knowledge base."

    lines = []
    for r in results:
        lines.append(f"[{r['title']}] (relevance: {r['score']})")
        lines.append(r["content"])
        lines.append("")
        # Empty string adds a blank line between documents for readability

    return "\n".join(lines)