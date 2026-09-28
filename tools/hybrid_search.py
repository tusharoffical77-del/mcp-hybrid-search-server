"""
Hybrid search: combines two different retrieval strategies over the same
document set, then fuses their rankings into one result list.

  1. SPARSE (keyword) retrieval — BM25. Good at exact term matches:
     if the query says "VPN", it strongly favors documents containing
     the literal word "VPN". Weak at synonyms/rephrasing.

  2. DENSE (semantic) retrieval — TF-IDF vectors reduced with SVD
     (this combination is called Latent Semantic Analysis / LSA).
     This projects documents and queries into a lower-dimensional space
     where documents about similar *topics* end up close together, even
     without exact word overlap. Weak at precise term matching compared
     to BM25.

  Note: a "real" dense retriever would use neural sentence embeddings
  (e.g. sentence-transformers). That was deliberately avoided here
  because it requires downloading a ~90MB+ model and pulls in PyTorch,
  a much heavier dependency — not worth it for a learning project. LSA
  is a legitimate, classical alternative that captures a meaningful
  chunk of the same "semantic similarity" benefit with a tiny footprint
  (scikit-learn only, no model download). See README for how to upgrade
  to real embeddings later if you want to push retrieval quality further.

  3. FUSION — Reciprocal Rank Fusion (RRF). Instead of trying to
     directly compare BM25 scores and cosine similarities (which live
     on completely different, incomparable scales), RRF combines
     *rankings*: a document's fused score is the sum of 1/(k + rank)
     across both result lists. This is simple, has no tunable weights
     to get wrong, and is a widely-used real technique (not a toy).
"""

import glob
import os

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "documents")

# Reciprocal Rank Fusion constant — standard default from the original
# RRF paper (Cormack et al.). Higher k flattens the influence of rank
# differences; this default works well without tuning.
RRF_K = 60

# Number of latent semantic dimensions for the LSA/SVD step. Small
# document collections don't support many meaningful dimensions —
# this is a reasonable default for a few dozen short documents.
SVD_COMPONENTS = 50


def _tokenize(text: str) -> list:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class HybridSearchIndex:
    """
    Loads all .txt files from documents/, builds both a BM25 index and
    an LSA (TF-IDF + SVD) index over them, and answers hybrid queries.

    Built once at server startup — rebuilding per-query would be wasteful
    since the document set doesn't change while the server is running.
    """

    def __init__(self, documents_dir: str = DOCUMENTS_DIR):
        self.doc_paths = sorted(glob.glob(os.path.join(documents_dir, "*.txt")))
        if not self.doc_paths:
            raise ValueError(f"No .txt documents found in {documents_dir}")

        self.doc_names = [os.path.basename(p) for p in self.doc_paths]
        self.doc_texts = []
        for path in self.doc_paths:
            with open(path, "r", encoding="utf-8") as f:
                self.doc_texts.append(f.read())

        # --- BM25 (sparse) index ---
        tokenized_docs = [_tokenize(t) for t in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)

        # --- LSA (dense-ish) index ---
        # min_df=1 since this is a small collection; a larger corpus
        # would normally filter out very rare terms.
        n_components = min(SVD_COMPONENTS, len(self.doc_texts) - 1, 100)
        n_components = max(n_components, 1)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.doc_vectors = self.svd.fit_transform(tfidf_matrix)

    def _bm25_ranking(self, query: str) -> list:
        """Returns doc indices ranked best-to-worst by BM25 score."""
        scores = self.bm25.get_scores(_tokenize(query))
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def _lsa_ranking(self, query: str) -> list:
        """Returns doc indices ranked best-to-worst by semantic similarity."""
        query_tfidf = self.vectorizer.transform([query])
        query_vector = self.svd.transform(query_tfidf)
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        return sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )

    def search(self, query: str, top_k: int = 3) -> list:
        """
        Run both retrievers, fuse via Reciprocal Rank Fusion, and return
        the top_k documents as a list of dicts with name, snippet, and
        fused score.
        """
        bm25_ranking = self._bm25_ranking(query)
        lsa_ranking = self._lsa_ranking(query)

        rrf_scores = {}
        for rank, doc_idx in enumerate(bm25_ranking):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (RRF_K + rank)
        for rank, doc_idx in enumerate(lsa_ranking):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (RRF_K + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in ranked[:top_k]:
            text = self.doc_texts[doc_idx].strip()
            results.append(
                {
                    "document": self.doc_names[doc_idx],
                    "score": round(score, 4),
                    "text": text,
                }
            )
        return results