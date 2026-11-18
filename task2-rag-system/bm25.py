
import re
import math
from collections import Counter

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b

        self.tokenized_docs = [
            re.findall(r"\b\w+\b", doc.get("text", doc.get("content", "")).lower())
            for doc in documents
        ]

        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)

        self.doc_freqs = []
        for doc in self.tokenized_docs:
            self.doc_freqs.append(Counter(doc))

        self.idf = {}
        total_docs = len(documents)

        all_terms = set(
            term for doc in self.doc_freqs for term in doc
        )

        for term in all_terms:
            df = sum(1 for doc in self.doc_freqs if term in doc)
            self.idf[term] = math.log(
                1 + (total_docs - df + 0.5) / (df + 0.5)
            )

    def search(self, query, top_k=5):
        query_terms = re.findall(r"\b\w+\b", query.lower())
        scores = []

        for i, doc in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_lengths[i]

            for term in query_terms:
                if term not in doc:
                    continue

                tf = doc[term]
                idf = self.idf.get(term, 0.0)

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avgdl
                )

                score += idf * numerator / denominator

            scores.append((score, i))

        ranked = sorted(scores, reverse=True)[:top_k]

        return [
            {
                **self.documents[i],
                "score": score
            }
            for score, i in ranked
        ]
