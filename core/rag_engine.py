import re
import math
import numpy as np
from collections import Counter
from typing import List, Tuple


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return [c for c in chunks if len(c.strip()) > 20]


def tokenize(text: str):
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def build_vocab(corpus: List[str]):
    vocab = sorted(set(
        token
        for doc in corpus
        for token in tokenize(doc)
    ))
    return {word: i for i, word in enumerate(vocab)}


def compute_idf(corpus: List[str], vocab_index):
    N = len(corpus)
    idf = {}

    for word in vocab_index:
        doc_count = sum(
            1 for doc in corpus
            if word in tokenize(doc)
        )
        idf[word] = math.log((N + 1) / (doc_count + 1)) + 1

    return idf


def vectorize(text: str, vocab_index, idf):
    tokens = tokenize(text)
    counts = Counter(tokens)
    total = len(tokens) or 1

    vec = np.zeros(len(vocab_index))

    for word, count in counts.items():
        if word in vocab_index:
            idx = vocab_index[word]
            tf = count / total
            vec[idx] = tf * idf[word]

    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


class RAGEngine:
    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.vocab_index = {}
        self.idf = {}

    def load_documents(self, text: str):
        self.chunks = chunk_text(text)

        self.vocab_index = build_vocab(self.chunks)
        self.idf = compute_idf(self.chunks, self.vocab_index)

        self.embeddings = [
            vectorize(chunk, self.vocab_index, self.idf)
            for chunk in self.chunks
        ]

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if not self.chunks:
            return []

        query_vec = vectorize(
            query,
            self.vocab_index,
            self.idf
        )

        scored = [
            (chunk, cosine_similarity(query_vec, emb))
            for chunk, emb in zip(self.chunks, self.embeddings)
        ]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def is_loaded(self) -> bool:
        return len(self.chunks) > 0