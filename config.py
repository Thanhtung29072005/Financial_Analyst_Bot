import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Qdrant configurations
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "qdrant_db")
COLLECTION_NAME = "vietnamese_laws"

# Embedding settings
# Provider: "huggingface" hoặc "cohere"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cohere")

# HuggingFace model (dùng khi EMBEDDING_PROVIDER = "huggingface")
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"

# Cohere embedding (dùng khi EMBEDDING_PROVIDER = "cohere")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_EMBEDDING_MODEL = "embed-multilingual-v3.0"  # 1024 chiều, hỗ trợ 100+ ngôn ngữ

# Text Splitting settings
# Tăng kích thước chunk để đọc các điều luật đầy đủ hơn
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# LLM Settings
# We use Groq's fast models. llama-3.3-70b-versatile or mixtral-8x7b-32768
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0.1 # Keep it low for precise factual retrieval
LLM_MAX_TOKENS = 2048

# Retrieval Settings
# Tăng số lượng chunk trả về để có nhiều ngữ cảnh luật hơn
RETRIEVER_K = 7        # giữ lại cho backward-compat
VECTOR_K = 25          # số docs lấy từ Qdrant vector search (cho hybrid)
BM25_K = 25            # số docs lấy từ BM25 keyword search
RRF_K = 60             # hằng số RRF theo paper (Cormack et al. 2009)
RERANK_TOP_N = 7
RERANK_MODEL = "rerank-multilingual-v3.0"

# BM25 Index path (pickle file, lưu cạnh qdrant_db/)
BM25_INDEX_PATH = os.path.join(os.path.dirname(__file__), "qdrant_db", "bm25_index.pkl")

# SQL Server configurations 
SQL_SERVER = os.getenv("SQL_SERVER", "localhost")
SQL_DATABASE = os.getenv("SQL_DATABASE", "Legal_Chatbot_DB")
SQL_TRUSTED_CONNECTION = os.getenv("SQL_TRUSTED_CONNECTION", "yes") # "yes" cho Windows Auth, "no" cho SQL Server Auth
SQL_USERNAME = os.getenv("SQL_USERNAME", "")
SQL_PASSWORD = os.getenv("SQL_PASSWORD", "")
