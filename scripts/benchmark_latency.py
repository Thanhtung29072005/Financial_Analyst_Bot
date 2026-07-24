"""
Benchmark Latency Script for Legal Advisory Chatbot
Đo lường thời gian xử lý chi tiết (Breakdown Latency) và tính các chỉ số P50, P95, Avg Latency.
"""

import os
import sys
import time
import json
import statistics
import pandas as pd
from dotenv import load_dotenv

# Reconfigure stdout/stderr to support Vietnamese characters on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"), override=True)

import config
from source.Function.search_Qdrant import FinancialRAG
from source.Generate.generate import rewrite_question, extract_entities_from_query, cohere_rerank, get_qa_chain, reciprocal_rank_fusion
from source.Function.utils import extract_law_name_from_filename
from qdrant_client.models import Filter, FieldCondition, MatchValue

def benchmark_single_query(rag_engine, question):
    """
    Đo thời gian từng công đoạn trong pipeline cho 1 câu hỏi.
    Returns: dict chứa timing breakdown (giây) và kết quả
    """
    t_start = time.perf_counter()

    # 1. Rewrite & Entity Extraction
    t0 = time.perf_counter()
    rewritten = rewrite_question(rag_engine, question, chat_history=[])
    entities = extract_entities_from_query(rag_engine, rewritten)
    t_rewrite = time.perf_counter() - t0

    # 2. Filtering & Search (Vector + BM25)
    t0 = time.perf_counter()
    source_filter = None
    if entities.get("law_name"):
        normalized_law_name = entities["law_name"].lower().replace(" ", "").replace("đ", "d")
        indexed_docs = rag_engine.get_indexed_documents()
        for doc_name in indexed_docs:
            normalized_doc = doc_name.lower().replace(" ", "")
            cleaned_law_name = extract_law_name_from_filename(doc_name).lower().replace(" ", "")
            if (normalized_law_name in normalized_doc
                    or normalized_doc in normalized_law_name
                    or normalized_law_name in cleaned_law_name
                    or cleaned_law_name in normalized_law_name):
                source_filter = doc_name
                break

    must_conditions = []
    if source_filter:
        must_conditions.append(FieldCondition(key="metadata.source", match=MatchValue(value=source_filter)))
    if entities.get("article"):
        must_conditions.append(FieldCondition(key="metadata.article", match=MatchValue(value=entities["article"])))
    if entities.get("chapter"):
        must_conditions.append(FieldCondition(key="metadata.chapter", match=MatchValue(value=entities["chapter"])))
    qdrant_filter = Filter(must=must_conditions) if must_conditions else None

    # Vector search
    vector_results = []
    initial_k = getattr(config, "VECTOR_K", 25)
    try:
        vector_results = rag_engine.vectorstore.similarity_search(
            query=rewritten,
            k=initial_k,
            filter=qdrant_filter
        )
    except Exception:
        pass

    if not vector_results and qdrant_filter:
        try:
            if source_filter:
                relaxed_filter = Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source_filter))])
                vector_results = rag_engine.vectorstore.similarity_search(query=rewritten, k=initial_k, filter=relaxed_filter)
            if not vector_results:
                vector_results = rag_engine.vectorstore.similarity_search(query=rewritten, k=initial_k)
        except Exception:
            pass

    # BM25 search
    bm25_results = []
    if hasattr(rag_engine, "bm25_store") and rag_engine.bm25_store:
        bm25_results = rag_engine.bm25_store.search(rewritten, k=getattr(config, "BM25_K", 25))

    # RRF Fusion
    results = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=getattr(config, "RRF_K", 60))
    t_search = time.perf_counter() - t0

    # 3. Cohere Rerank
    t0 = time.perf_counter()
    reranked = cohere_rerank(
        query=rewritten,
        documents=results,
        cohere_api_key=config.COHERE_API_KEY,
        top_n=config.RERANK_TOP_N
    )
    t_rerank = time.perf_counter() - t0

    # 4. LLM Generation
    t0 = time.perf_counter()
    qa_chain = get_qa_chain(rag_engine)
    answer = qa_chain.invoke({
        "input": rewritten,
        "chat_history": [],
        "context": reranked
    })
    t_llm = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start

    return {
        "question": question,
        "t_rewrite": round(t_rewrite, 3),
        "t_search": round(t_search, 3),
        "t_rerank": round(t_rerank, 3),
        "t_retrieval_total": round(t_rewrite + t_search + t_rerank, 3),
        "t_llm": round(t_llm, 3),
        "t_total": round(t_total, 3),
        "answer_len": len(answer)
    }


def calculate_p95(data):
    """Tính giá trị P95 (95th percentile)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(0.95 * (len(sorted_data) - 1))
    return sorted_data[idx]


def main():
    print("=" * 65)
    print("🚀 BẮT ĐẦU BENCHMARK LATENCY HYBRID RAG SYSTEM")
    print("=" * 65)

    dataset_path = os.path.join(ROOT_DIR, "data", "eval_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"[!] Không tìm thấy eval dataset tại {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"[*] Tải {len(eval_data)} câu hỏi kiểm thử.")
    print("[*] Đang khởi tạo RAG Engine...")

    rag_engine = FinancialRAG()
    if not rag_engine.load_existing_db():
        print("[!] Không thể load Vectorstore.")
        return

    print("\n[*] Đang chạy benchmark...")
    records = []

    # Chạy warm-up 1 câu (để load model/connection)
    print("[*] Warm-up pipeline...")
    try:
        benchmark_single_query(rag_engine, "Đất đai thuộc sở hữu của ai?")
    except Exception:
        pass

    for idx, item in enumerate(eval_data):
        q = item["question"]
        print(f"  [{idx+1}/{len(eval_data)}] Testing: '{q[:40]}...' ", end="", flush=True)
        try:
            res = benchmark_single_query(rag_engine, q)
            records.append(res)
            print(f"-> Total: {res['t_total']}s (Retrieval: {res['t_retrieval_total']}s | LLM: {res['t_llm']}s)")
        except Exception as e:
            print(f"-> [!] Lỗi: {e}")

        if idx < len(eval_data) - 1:
            time.sleep(10)  # Nghỉ 10s giữa các câu hỏi để tránh Groq API Rate Limit retry làm ảo chỉ số

    if not records:
        print("[!] Không có kết quả benchmark.")
        return

    df = pd.DataFrame(records)

    # Tính thống kê
    totals = df["t_total"].tolist()
    retrievals = df["t_retrieval_total"].tolist()
    llms = df["t_llm"].tolist()

    p50_total = statistics.median(totals)
    p95_total = calculate_p95(totals)
    avg_total = statistics.mean(totals)

    avg_retrieval = statistics.mean(retrievals)
    avg_llm = statistics.mean(llms)

    print("\n" + "=" * 65)
    print("📊 BẢNG KẾT QUẢ BENCHMARK LATENCY (CHI TIẾT)")
    print("=" * 65)

    print(f"\n1. Tổng thời gian phản hồi End-to-End (Latency):")
    print(f"   • P50 (Trung vị - Median) : {p50_total:.3f} s")
    print(f"   • P95 (95th Percentile)   : {p95_total:.3f} s")
    print(f"   • Trung bình (Average)    : {avg_total:.3f} s")

    print(f"\n2. Phân rã thời gian xử lý (Breakdown Average):")
    print(f"   • Entity & Search (BM25+Vector) : {avg_retrieval:.3f} s ({avg_retrieval/avg_total*100:.1f}%)")
    print(f"   • LLM Answer Generation (Groq)   : {avg_llm:.3f} s ({avg_llm/avg_total*100:.1f}%)")

    print("\n" + "=" * 65)
    print("📝 CÂU VIẾT CHO CV:")
    print("-" * 65)
    print(f"\"Cut P50 end-to-end latency to {p50_total:.2f}s (P95: {p95_total:.2f}s) by employing hybrid vector+BM25 retrieval with fast local RRF fusion and lightweight model entity extraction.\"")
    print("=" * 65)

    # Export CSV
    output_csv = os.path.join(ROOT_DIR, "data", "latency_benchmark.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[*] Kết quả chi tiết đã xuất ra: {output_csv}")


if __name__ == "__main__":
    main()
