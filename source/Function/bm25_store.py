"""
BM25Store — Quản lý BM25 keyword search index cho Legal Advisory Chatbot.

Dùng thư viện `rank-bm25` (BM25Okapi) để index và tìm kiếm văn bản pháp luật.
Index được persist ra disk bằng pickle để tránh rebuild mỗi lần restart server.
"""

import os
import re
import pickle
import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _tokenize_vi(text: str) -> List[str]:
    """
    Tokenizer đơn giản cho tiếng Việt:
    - Lowercase
    - Tách theo khoảng trắng và dấu câu
    - Giữ lại các token có độ dài >= 1
    """
    text = text.lower()
    # Tách theo ký tự không phải chữ cái/số (bao gồm dấu câu tiếng Việt)
    tokens = re.split(r'[^\w]+', text, flags=re.UNICODE)
    return [t for t in tokens if len(t) >= 1]


class BM25Store:
    """
    Wrapper quản lý BM25 index trên tập documents.

    Attributes:
        _docs:      List[Document] — tất cả documents đã được index
        _tokenized: List[List[str]] — tokenized version của từng doc
        _bm25:      BM25Okapi object (lazy-rebuilt khi index thay đổi)
        _dirty:     bool — True nếu index cần rebuild
    """

    def __init__(self):
        self._docs: List[Document] = []
        self._tokenized: List[List[str]] = []
        self._bm25 = None
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self):
        """Rebuild BM25Okapi index từ danh sách documents hiện tại."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Thư viện 'rank-bm25' chưa được cài. Chạy: pip install rank-bm25"
            )

        if not self._tokenized:
            self._bm25 = None
            self._dirty = False
            return

        self._bm25 = BM25Okapi(self._tokenized)
        self._dirty = False
        logger.info(f"[BM25Store] Rebuilt index với {len(self._docs)} documents.")

    def _ensure_index(self):
        """Đảm bảo index luôn up-to-date trước khi search."""
        if self._dirty or self._bm25 is None:
            self._rebuild_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, docs: List[Document]):
        """
        Thêm danh sách Document vào BM25 index.
        Index sẽ được rebuild lazy (chỉ khi search được gọi).
        """
        for doc in docs:
            tokens = _tokenize_vi(doc.page_content)
            self._docs.append(doc)
            self._tokenized.append(tokens)
        self._dirty = True
        logger.info(f"[BM25Store] Đã thêm {len(docs)} docs. Tổng: {len(self._docs)}.")

    def delete_by_source(self, file_name: str) -> int:
        """
        Xóa tất cả documents có metadata['source'] == file_name khỏi BM25 index.

        Returns:
            Số documents đã xóa.
        """
        original_count = len(self._docs)
        filtered = [
            (doc, tok)
            for doc, tok in zip(self._docs, self._tokenized)
            if doc.metadata.get("source") != file_name
        ]
        self._docs = [doc for doc, _ in filtered]
        self._tokenized = [tok for _, tok in filtered]

        removed = original_count - len(self._docs)
        if removed > 0:
            self._dirty = True
            logger.info(f"[BM25Store] Đã xóa {removed} docs của '{file_name}'.")
        return removed

    def search(self, query: str, k: int = 25) -> List[Document]:
        """
        Tìm kiếm top-k documents theo BM25 score.

        Args:
            query: Câu hỏi / từ khóa tìm kiếm.
            k:     Số lượng kết quả tối đa trả về.

        Returns:
            List[Document] được sắp xếp theo BM25 score giảm dần.
        """
        if not self._docs:
            return []

        self._ensure_index()

        if self._bm25 is None:
            return []

        query_tokens = _tokenize_vi(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Lấy top-k index có score > 0
        scored_indices = [
            (i, score) for i, score in enumerate(scores) if score > 0
        ]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_k = scored_indices[:k]

        results = []
        for idx, score in top_k:
            doc = self._docs[idx]
            # Gắn BM25 score vào metadata để có thể debug
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": round(float(score), 4)}
            )
            results.append(enriched_doc)

        logger.debug(f"[BM25Store] Query '{query[:50]}...' → {len(results)} kết quả.")
        return results

    def __len__(self) -> int:
        return len(self._docs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """
        Lưu BM25Store ra file pickle.
        Chỉ lưu _docs và _tokenized (không lưu BM25Okapi object vì có thể rebuild).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "docs": self._docs,
            "tokenized": self._tokenized,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[BM25Store] Đã lưu {len(self._docs)} docs vào '{path}'.")

    def load(self, path: str) -> bool:
        """
        Load BM25Store từ file pickle.

        Returns:
            True nếu load thành công, False nếu file không tồn tại hoặc lỗi.
        """
        if not os.path.exists(path):
            logger.info(f"[BM25Store] Không tìm thấy index tại '{path}'. Bắt đầu fresh.")
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._docs = state.get("docs", [])
            self._tokenized = state.get("tokenized", [])
            self._dirty = True  # Cần rebuild BM25Okapi object
            logger.info(f"[BM25Store] Đã load {len(self._docs)} docs từ '{path}'.")
            return True
        except Exception as e:
            logger.warning(f"[BM25Store] Lỗi khi load index: {e}. Reset về empty.")
            self._docs = []
            self._tokenized = []
            self._bm25 = None
            return False

    @classmethod
    def from_file(cls, path: str) -> "BM25Store":
        """Factory method: tạo BM25Store và load từ file (nếu có)."""
        store = cls()
        store.load(path)
        return store
