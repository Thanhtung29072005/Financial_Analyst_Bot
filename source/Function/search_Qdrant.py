import os
import sys

# Ensure parent directory is in path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from source.Function.utils import split_by_articles, extract_law_name_from_filename
from source.Function.bm25_store import BM25Store

class FinancialRAG:
    def __init__(self):
        from source.LLM.groq import get_llm, get_embeddings, get_qdrant_client
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.client = get_qdrant_client()
        self.vectorstore = None
        # BM25 store — load từ disk nếu có, hoặc bắt đầu mới
        self.bm25_store = BM25Store.from_file(config.BM25_INDEX_PATH)

    def load_existing_db(self):
        """Loads an existing Qdrant DB if it exists."""
        if os.path.exists(config.QDRANT_PATH):
            try:
                self.client.get_collection(config.COLLECTION_NAME)
                self.vectorstore = QdrantVectorStore(
                    client=self.client, 
                    collection_name=config.COLLECTION_NAME, 
                    embedding=self.embeddings
                )
                # BM25 index đã được load trong __init__ — log trạng thái
                print(f"[*] BM25 index: {len(self.bm25_store)} documents.")
                return True
            except Exception:
                pass
        return False

    def load_and_index_pdf(self, file_path, session_id=None):
        """Loads a PDF, splits it, and stores it in Qdrant Local."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        file_name = os.path.basename(file_path)
        # Trích xuất tên luật từ tên file để bổ sung metadata
        law_name = extract_law_name_from_filename(file_name)

        splits = split_by_articles(docs, law_name=law_name)
        is_article_split = splits is not None

        if not is_article_split:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            splits = text_splitter.split_documents(docs)

        for doc in splits:
            doc.metadata["source"] = file_name
            # Đảm bảo law_name luôn có trong metadata (kể cả fallback splitter)
            if "law_name" not in doc.metadata or not doc.metadata["law_name"]:
                doc.metadata["law_name"] = law_name
            if "title" not in doc.metadata or not doc.metadata["title"]:
                doc.metadata["title"] = law_name
            if session_id is not None:
                doc.metadata["session_id"] = session_id

        try:
            self.client.get_collection(config.COLLECTION_NAME)
        except Exception:
            from qdrant_client.models import VectorParams, Distance
            vector_size = len(self.embeddings.embed_query("test"))
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            self.client.delete(
                collection_name=config.COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=file_name)
                        )
                    ]
                )
            )
        except Exception:
            pass

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=config.COLLECTION_NAME,
            embedding=self.embeddings
        )
        self.vectorstore.add_documents(splits)

        # Thêm vào BM25 index (xóa docs cũ của file này trước để tránh duplicate)
        self.bm25_store.delete_by_source(file_name)
        self.bm25_store.add_documents(splits)
        self.bm25_store.save(config.BM25_INDEX_PATH)
        print(f"[*] BM25 index cập nhật: {len(self.bm25_store)} docs tổng cộng.")

        return len(splits)

    def delete_document(self, file_name):
        """Xóa hoàn toàn một tài liệu khỏi cơ sở dữ liệu Qdrant, BM25 index và thư mục data/laws."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            self.client.delete(
                collection_name=config.COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=file_name)
                        )
                    ]
                )
            )
            # Đồng bộ xóa khỏi BM25 index
            removed = self.bm25_store.delete_by_source(file_name)
            if removed > 0:
                self.bm25_store.save(config.BM25_INDEX_PATH)

            file_path = os.path.join("data", "laws", file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception:
            return False

    def get_indexed_documents(self):
        """Lấy danh sách các tài liệu luật (tên file PDF) đã được nạp vào Qdrant."""
        if not self.vectorstore:
            return []
        try:
            sources = set()
            next_page_offset = None
            while True:
                response, next_page_offset = self.client.scroll(
                    collection_name=config.COLLECTION_NAME,
                    limit=1000,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False
                )
                for point in response:
                    payload = point.payload
                    # pyrefly: ignore [missing-attribute]
                    metadata = payload.get("metadata", {})
                    # pyrefly: ignore [missing-attribute]
                    source = metadata.get("source") or payload.get("source")
                    if source:
                        sources.add(source)
                if next_page_offset is None:
                    break
            return sorted(list(sources))
        except Exception:
            return []

    def rewrite_question(self, question, chat_history):
        from source.Generate.generate import rewrite_question
        return rewrite_question(self, question, chat_history)

    def extract_entities_from_query(self, query):
        from source.Generate.generate import extract_entities_from_query
        return extract_entities_from_query(self, query)

    def get_qa_chain(self):
        from source.Generate.generate import get_qa_chain
        return get_qa_chain(self)

    def summarize_pdf(self, file_path):
        from source.Generate.T5Sum import summarize_pdf
        return summarize_pdf(self, file_path)

    def ask(self, question, chat_history, session_id=None):
        from source.Generate.generate import ask
        return ask(self, question, chat_history, session_id)


