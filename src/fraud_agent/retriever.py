"""Policy retrieval: Chroma vector store with OpenAI embeddings, or an offline TF-IDF index."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _chunks(policies_dir: Path) -> list[dict[str, str]]:
    """Split each policy by '## ' / '**Rule' paragraphs so retrieval returns focused sections."""
    out = []
    for path in sorted(policies_dir.glob("*.md")):
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and not p.startswith("# ")]
        for i, para in enumerate(paragraphs):
            out.append({"id": f"{path.stem}#{i}", "doc": path.name, "title": title,
                        "text": f"{title}: {para}"})
    return out


class TfidfRetriever:
    def __init__(self, policies_dir: Path):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = _chunks(policies_dir)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        sims = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        top = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": round(float(sims[i]), 4)} for i in top if sims[i] > 0]


class ChromaRetriever:
    def __init__(self, policies_dir: Path, persist_dir: Path, embedding_model: str):
        import chromadb
        from chromadb.utils import embedding_functions

        ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"],
                                                         model_name=embedding_model)
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.col = client.get_or_create_collection("policies", embedding_function=ef,
                                                   metadata={"hnsw:space": "cosine"})
        chunks = _chunks(policies_dir)
        if self.col.count() != len(chunks):
            existing = self.col.get()["ids"]
            if existing:
                self.col.delete(ids=existing)
            self.col.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
                         metadatas=[{"doc": c["doc"], "title": c["title"]} for c in chunks])

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        res = self.col.query(query_texts=[query], n_results=k)
        return [{"id": i, "doc": m["doc"], "title": m["title"], "text": d, "score": round(1 - dist, 4)}
                for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                         res["distances"][0])]


def build_retriever(kind: str, policies_dir: Path, persist_dir: Path, embedding_model: str):
    if kind == "chroma" and os.getenv("OPENAI_API_KEY"):
        return ChromaRetriever(policies_dir, persist_dir, embedding_model)
    return TfidfRetriever(policies_dir)
