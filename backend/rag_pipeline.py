"""Build a text knowledge base from PDFs for RAG retrieval.

Reads PDFs from a raw docs directory, chunks pages by word count with overlap,
and writes a JSONL knowledge base used for retrieval (e.g. BM25 or vector search).
"""
from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np


RAW_DOCS_DIR = Path("data/data/docs/raw")
PROCESSED_DIR = Path("data/data/processed")
KB_PATH = PROCESSED_DIR / "kb.jsonl"
INDEX_DIR = Path("data/data/index")
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_INDEX_PATH = INDEX_DIR / "embeddings.npy"
EMBED_META_PATH = INDEX_DIR / "embeddings_meta.json"
MISSING_EVIDENCE_THRESHOLD = 0.20

_MODEL: Optional[SentenceTransformer] = None


def get_model(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    """Return the shared SentenceTransformer instance; load once and reuse (singleton)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


_KB_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def load_kb_map(kb_path: Path = KB_PATH) -> Dict[str, Dict[str, Any]]:
    """Load the knowledge base into a map keyed by evidence_id; cached after first load."""
    global _KB_CACHE
    if _KB_CACHE is not None:
        return _KB_CACHE

    m: Dict[str, Dict[str, Any]] = {}
    with kb_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            m[obj["evidence_id"]] = obj
    _KB_CACHE = m
    return m


_WORD_RE = re.compile(r"\S+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "what", "are", "is", "was", "were",
    "be", "as", "by", "from", "at", "it", "this", "that", "these", "those", "do", "does", "did", "how", "why",
}


def _keywords(text: str) -> Set[str]:
    """Extract meaningful query keywords (alpha tokens of length >= 3, excluding stopwords)."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def evidence_supports_query(question: str, evidence_texts: List[str], min_hits: int = 2) -> bool:
    """Return True if at least min_hits query keywords appear in the combined evidence text (topic-match gate).

    Deterministic, no LLM. Prevents generic/loosely related retrieval from being treated as valid evidence.
    """
    qk = _keywords(question)
    if not qk:
        return True
    joined = " ".join(evidence_texts).lower()
    hits = sum(1 for kw in qk if kw in joined)
    return hits >= min_hits


def pick_best_answer_evidence(question: str, evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose the evidence item that best matches the question by keyword overlap, then hybrid_score tie-break."""
    qk = _keywords(question)
    if not evidence or not qk:
        return evidence[0] if evidence else None

    def overlap(e: Dict[str, Any]) -> int:
        t = (e.get("text") or "").lower()
        return sum(1 for kw in qk if kw in t)

    return max(evidence, key=lambda e: (overlap(e), e.get("hybrid_score", 0.0)))


@dataclass(frozen=True)
class Chunk:
    """A single text chunk from a PDF, with source and position metadata.

    Attributes:
        evidence_id: Unique id, e.g. ``{file_stem}_p{page:02d}_c{chunk:03d}``.
        source_file: Original PDF filename.
        page: 1-based page number.
        chunk_index: 1-based index of this chunk within the page.
        text: Chunk text content.
    """
    evidence_id: str
    source_file: str
    page: int
    chunk_index: int
    text: str


def _clean_text(s: str) -> str:
    """Normalize whitespace and remove null bytes for consistent chunking.

    Args:
        s: Raw text, possibly with extra whitespace or nulls.

    Returns:
        Text with nulls replaced by spaces and runs of whitespace collapsed to single spaces, trimmed.
    """
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _chunk_by_words(text: str, chunk_words: int = 250, overlap_words: int = 40) -> List[str]:
    """Split text into fixed-size word windows with overlap for retrieval-friendly chunks.

    Args:
        text: Input text to chunk (words are any non-whitespace sequences).
        chunk_words: Target number of words per chunk.
        overlap_words: Number of words to overlap between consecutive chunks.

    Returns:
        List of chunk strings; empty list if text has no words.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def build_kb(
    raw_dir: Path = RAW_DOCS_DIR,
    kb_path: Path = KB_PATH,
    chunk_words: int = 250,
    overlap_words: int = 40,
) -> List[Chunk]:
    """Build the knowledge base from PDFs and write it to JSONL.

    Scans ``raw_dir`` for ``*.pdf`` files, extracts text per page, chunks by word count
    with overlap, assigns evidence IDs, and writes one JSON object per chunk to ``kb_path``.

    Args:
        raw_dir: Directory containing PDF files. Defaults to ``RAW_DOCS_DIR``.
        kb_path: Output path for the JSONL knowledge base. Parent directory is created if needed.
        chunk_words: Target words per chunk. Default 250.
        overlap_words: Word overlap between consecutive chunks. Default 40.

    Returns:
        List of all created ``Chunk`` instances in order.

    Raises:
        FileNotFoundError: If ``raw_dir`` contains no PDF files.
    """
    raw_dir = Path(raw_dir)
    kb_path = Path(kb_path)
    kb_path.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in raw_dir.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")

    all_chunks: List[Chunk] = []

    for pdf_path in pdfs:
        reader = PdfReader(str(pdf_path))
        file_stem = pdf_path.stem

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = _clean_text(page_text)
            if not page_text:
                continue

            page_chunks = _chunk_by_words(page_text, chunk_words=chunk_words, overlap_words=overlap_words)

            for ci, chunk_text in enumerate(page_chunks, start=1):
                evidence_id = f"{file_stem}_p{page_idx:02d}_c{ci:03d}"
                all_chunks.append(
                    Chunk(
                        evidence_id=evidence_id,
                        source_file=pdf_path.name,
                        page=page_idx,
                        chunk_index=ci,
                        text=chunk_text,
                    )
                )

    with kb_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(
                json.dumps(
                    {
                        "evidence_id": c.evidence_id,
                        "source_file": c.source_file,
                        "page": c.page,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return all_chunks


def build_bm25_index(kb_path: Path = KB_PATH) -> None:
    """Build the BM25 (sparse) index from the knowledge base and save it to disk.

    Reads chunk texts from ``kb_path``, tokenizes with lowercase split, builds
    a BM25Okapi index, and pickles it with evidence IDs to ``BM25_INDEX_PATH``.

    Args:
        kb_path: Path to the JSONL knowledge base. Defaults to ``KB_PATH``.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    texts = []
    evidence_ids = []

    with kb_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            evidence_ids.append(obj["evidence_id"])
            texts.append(obj["text"])

    tokenized = [t.lower().split() for t in texts]

    bm25 = BM25Okapi(tokenized)

    with BM25_INDEX_PATH.open("wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "evidence_ids": evidence_ids,
            },
            f,
        )

    print(f"BM25 index built with {len(evidence_ids)} documents")


def load_bm25_index(path: Path = BM25_INDEX_PATH):
    """Load the serialized BM25 index and evidence ID list from disk.

    Args:
        path: Path to the pickle file. Defaults to ``BM25_INDEX_PATH``.

    Returns:
        Tuple of ``(bm25, evidence_ids)``: the BM25Okapi instance and list of evidence IDs in index order.
    """
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["bm25"], payload["evidence_ids"]


def retrieve_sparse(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Retrieve top-k chunks using the BM25 (sparse) index.

    Args:
        query: Free-text query string.
        top_k: Number of results to return. Default 5.

    Returns:
        List of ``(evidence_id, score)`` tuples, sorted by score descending. Backend/UI friendly.
    """
    bm25, evidence_ids = load_bm25_index()

    q_tokens = query.lower().split()
    scores = bm25.get_scores(q_tokens)

    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [(evidence_ids[i], float(scores[i])) for i in top_idx]


def build_dense_index(kb_path: Path = KB_PATH, model_name: str = EMBED_MODEL_NAME) -> None:
    """Build the dense (embedding) index from the knowledge base and save to disk.

    Loads chunk texts from ``kb_path``, encodes with the given sentence-transformers
    model (normalized for cosine = dot product), and saves vectors to ``EMBED_INDEX_PATH``
    and metadata (model name, evidence_ids) to ``EMBED_META_PATH``.

    Args:
        kb_path: Path to the JSONL knowledge base. Defaults to ``KB_PATH``.
        model_name: Sentence-transformers model name. Defaults to ``EMBED_MODEL_NAME``.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    evidence_ids = []
    texts = []

    with kb_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            evidence_ids.append(obj["evidence_id"])
            texts.append(obj["text"])

    model = get_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    vectors = np.asarray(vectors, dtype=np.float32)
    np.save(EMBED_INDEX_PATH, vectors)

    with EMBED_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {"model_name": model_name, "evidence_ids": evidence_ids},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Dense index built: {vectors.shape} -> {EMBED_INDEX_PATH}")


def load_dense_index() -> Tuple[np.ndarray, List[str], str]:
    """Load the dense index vectors and metadata from disk.

    Returns:
        Tuple of ``(vectors, evidence_ids, model_name)``: embedding array, evidence IDs in order, and model name used.
    """
    vectors = np.load(EMBED_INDEX_PATH)
    with EMBED_META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return vectors, meta["evidence_ids"], meta["model_name"]


def retrieve_dense(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Retrieve top-k chunks using the dense (embedding) index.

    Args:
        query: Free-text query string.
        top_k: Number of results to return. Default 5.

    Returns:
        List of ``(evidence_id, score)`` tuples, sorted by score descending. Scores are cosine similarity (dot product).
    """
    vectors, evidence_ids, model_name = load_dense_index()
    model = get_model(model_name)

    q_vec = model.encode([query], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype=np.float32)[0]

    scores = vectors @ q_vec

    top_idx = np.argsort(-scores)[:top_k]
    return [(evidence_ids[i], float(scores[i])) for i in top_idx]


def _minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores to [0, 1] by min-max. Used for hybrid fusion.

    Args:
        scores: Map of evidence_id -> raw score.

    Returns:
        Map of evidence_id -> score in [0, 1]. If all scores equal, returns 1.0 for each.
    """
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


def retrieve_sparse_topn(query: str, top_n: int = 50) -> Dict[str, float]:
    """Retrieve top-n chunks from BM25 as a score map (for hybrid fusion).

    Args:
        query: Free-text query string.
        top_n: Number of candidates to return. Default 50.

    Returns:
        Dict mapping evidence_id -> BM25 score for the top-n chunks.
    """
    bm25, evidence_ids = load_bm25_index()
    q_tokens = query.lower().split()
    scores = bm25.get_scores(q_tokens)

    top_idx = np.argsort(-scores)[:top_n]
    return {evidence_ids[i]: float(scores[i]) for i in top_idx}


def retrieve_dense_topn(query: str, top_n: int = 50) -> Dict[str, float]:
    """Retrieve top-n chunks from the dense index as a score map (for hybrid fusion).

    Args:
        query: Free-text query string.
        top_n: Number of candidates to return. Default 50.

    Returns:
        Dict mapping evidence_id -> similarity score for the top-n chunks.
    """
    vectors, evidence_ids, model_name = load_dense_index()
    model = get_model(model_name)
    q_vec = model.encode([query], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype=np.float32)[0]
    scores = vectors @ q_vec

    top_idx = np.argsort(-scores)[:top_n]
    return {evidence_ids[i]: float(scores[i]) for i in top_idx}


def retrieve_hybrid(
    query: str,
    top_k: int = 5,
    alpha: float = 0.6,
    candidate_pool: int = 50,
) -> List[Tuple[str, float, float, float]]:
    """Retrieve top-k chunks using hybrid (sparse + dense) fusion.

    Fetches top-n candidates from both BM25 and dense retrieval, min-max normalizes
    scores, then blends with ``(1 - alpha) * sparse + alpha * dense`` and returns top-k.

    Args:
        query: Free-text query string.
        top_k: Number of final results to return. Default 5.
        alpha: Weight for dense score in [0, 1]; (1 - alpha) for sparse. Default 0.6.
        candidate_pool: Number of candidates to take from each retriever before fusion. Default 50.

    Returns:
        List of ``(evidence_id, hybrid_score, bm25_norm, dense_norm)`` tuples, sorted by hybrid_score descending.
    """
    sparse_raw = retrieve_sparse_topn(query, top_n=candidate_pool)
    dense_raw = retrieve_dense_topn(query, top_n=candidate_pool)

    sparse_norm = _minmax_norm(sparse_raw)
    dense_norm = _minmax_norm(dense_raw)

    all_ids = set(sparse_norm.keys()) | set(dense_norm.keys())

    blended = []
    for eid in all_ids:
        s = sparse_norm.get(eid, 0.0)
        d = dense_norm.get(eid, 0.0)
        h = (1 - alpha) * s + alpha * d
        blended.append((eid, h, s, d))

    blended.sort(key=lambda x: x[1], reverse=True)
    return blended[:top_k]


def run_hybrid_query(
    question: str,
    top_k: int = 5,
    alpha: float = 0.6,
    candidate_pool: int = 50,
) -> Dict[str, Any]:
    """Single entry point for hybrid RAG: ensures indexes, runs retrieval, returns evidence with citations and metadata.

    Backend teammate: call run_hybrid_query(question, top_k, alpha), return the JSON as-is. Log latency_ms,
    missing_evidence_behavior, and top evidence IDs. UI can display evidence[].citation and evidence[].text directly.

    Args:
        question: User query string.
        top_k: Number of evidence items to return. Default 5.
        alpha: Dense weight for hybrid fusion. Default 0.6.
        candidate_pool: Candidates per retriever before fusion. Default 50.

    Returns:
        Dict with keys: question, top_k, alpha, latency_ms, evidence (list with evidence_id, hybrid_score,
        bm25_norm, dense_norm, source_file, page, text snippet, citation), missing_evidence_behavior,
        support_gate_pass (True if query keywords appear in evidence), answer.
    """
    ensure_indexes()
    t0 = time.time()

    hits = retrieve_hybrid(question, top_k=top_k, alpha=alpha, candidate_pool=candidate_pool)
    kb = load_kb_map()

    evidence = []
    for eid, h, s, d in hits:
        item = kb.get(eid, {})
        citation = f"({item.get('source_file', '?')} p{item.get('page', '?')}, chunk {item.get('chunk_index', '?')})"
        evidence.append({
            "evidence_id": eid,
            "hybrid_score": h,
            "bm25_norm": s,
            "dense_norm": d,
            "source_file": item.get("source_file"),
            "page": item.get("page"),
            "text": (item.get("text") or "")[:600],
            "citation": citation,
        })

    evidence_texts = [e["text"] for e in evidence]
    qk = _keywords(question)
    min_hits = 1 if len(qk) <= 3 else 2
    supported = evidence_supports_query(question, evidence_texts, min_hits=min_hits)
    missing = (not hits) or (not supported)
    best = pick_best_answer_evidence(question, evidence)
    if missing:
        answer = "Not enough evidence in the retrieved context."
    else:
        answer = (best["text"] if best else evidence[0]["text"])

    latency_ms = (time.time() - t0) * 1000
    return {
        "question": question,
        "top_k": top_k,
        "alpha": alpha,
        "latency_ms": latency_ms,
        "evidence": evidence,
        "missing_evidence_behavior": missing,
        "support_gate_pass": supported,
        "answer": answer,
    }

def gold_label_helper(query: str, top_k: int = 10, alpha: float = 0.6):
    res = run_hybrid_query(query, top_k=top_k, alpha=alpha)
    for i, e in enumerate(res["evidence"], start=1):
        print(f"{i}. {e['evidence_id']}  {e['citation']}  hybrid={e['hybrid_score']:.3f}")
        print(f"   {e['text'][:220]}...\n")


def ensure_indexes() -> None:
    """Build KB and BM25/dense indexes if they do not already exist."""
    if not KB_PATH.exists():
        build_kb()

    if not BM25_INDEX_PATH.exists():
        build_bm25_index()

    if not EMBED_INDEX_PATH.exists() or not EMBED_META_PATH.exists():
        build_dense_index()

if __name__ == "__main__":
    ensure_indexes()

    test_queries = [
        # BM25 PRF (should hit bm25_prf.pdf)
        "Explain BM25 length normalization and the difference between verbosity vs scope hypotheses.",
        "What is the eliteness model and how does it relate to BM25?",
        "What is BM25F and what does it change compared to BM25?",

        # RAG evaluation (should hit rag_evaluation_ragas.pdf / benchmark paper)
        "What metrics does RAGAS use to evaluate a RAG system (faithfulness, answer relevance, context relevance), and what do they mean?",
        "What does faithfulness mean in RAG evaluation and why does it matter?",

        # Chain-of-Retrieval (should hit chain_of_retrieval.pdf)
        "What is Chain-of-Retrieval Augmented Generation and why does multi-step retrieval help?",
        "Does Chain-of-Retrieval always help? What happens as chain length increases?",

        # GraphFlow (should hit graphflow_rag.pdf)
        "How does GraphFlow improve retrieval-augmented generation compared to vanilla RAG?",
        "What benchmark does GraphFlow evaluate on and what is the main claim?",

        # GFM-RAG (should hit gfm_rag.pdf)
        "What is GFM-RAG and what problem is it trying to solve?",

        # Off-topic / unanswerable (should be missing=True)
        "What are the economic impacts of climate change?",
        "Who won the 2024 Super Bowl?",
    ]

    for q in test_queries:
        res = run_hybrid_query(q, top_k=5, alpha=0.6)

        print("\n" + "=" * 90)
        print("Q:", res["question"])
        print(f"missing={res['missing_evidence_behavior']}  support_gate={res['support_gate_pass']}  latency_ms={res['latency_ms']:.2f}")
        print("Answer:", res["answer"][:250].replace("\n", " "), "...")
        print("\nTop evidence:")
        for i, e in enumerate(res["evidence"], 1):
            print(f"  {i}. {e['citation']}  h={e['hybrid_score']:.3f}  bm25={e['bm25_norm']:.3f}  dense={e['dense_norm']:.3f}")
            print(f"     {e['text'][:180].replace('\\n',' ')}...")




