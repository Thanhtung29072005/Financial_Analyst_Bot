import os
import sys

# Reconfigure stdout/stderr to support Vietnamese characters on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        # pyrefly: ignore [missing-attribute]
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        # pyrefly: ignore [missing-attribute]
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import json
import pandas as pd
from dotenv import load_dotenv

# Thêm thư mục gốc (project root) vào sys.path để import config và source
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# Load environment variables
load_dotenv(os.path.join(ROOT_DIR, ".env"), override=True)

import config
from source.Function.search_Qdrant import FinancialRAG
from source.Generate.generate import rewrite_question, extract_entities_from_query, cohere_rerank, get_qa_chain
from source.Function.utils import extract_law_name_from_filename
from qdrant_client.models import Filter, FieldCondition, MatchValue
from datasets import Dataset
from ragas import evaluate, RunConfig
try:
    # RAGAS >= 0.2: metrics phải được khởi tạo như objects
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    RAGAS_METRICS = [
        Faithfulness(),
        # strictness=1: Groq không hỗ trợ n>1 trong API call
        # RAGAS mặc định strictness=3 sẽ gây BadRequestError trên Groq
        AnswerRelevancy(strictness=1),
        ContextPrecision(),
        ContextRecall(),
    ]
except ImportError:
    # Fallback cho RAGAS cũ dùng singleton
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    RAGAS_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

def query_rag_pipeline(rag_engine, question):
    """
    Runs a query through the full Hybrid RAG pipeline (Vector + BM25 + RRF + Rerank) to gather:
    - answer: The final generated response.
    - contexts: A list of raw text contents (page_content) of the retrieved/reranked documents.
    """
    # 1. Rewrite question
    rewritten_question = rewrite_question(rag_engine, question, chat_history=[])
    
    # 2. Extract entities
    entities = extract_entities_from_query(rag_engine, rewritten_question)
    
    # 3. Map law_name to document filename
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
                
    # 4. Qdrant filters
    must_conditions = []
    if source_filter:
        must_conditions.append(FieldCondition(key="metadata.source", match=MatchValue(value=source_filter)))
    if entities.get("article"):
        must_conditions.append(FieldCondition(key="metadata.article", match=MatchValue(value=entities["article"])))
    if entities.get("chapter"):
        must_conditions.append(FieldCondition(key="metadata.chapter", match=MatchValue(value=entities["chapter"])))
        
    qdrant_filter = Filter(must=must_conditions) if must_conditions else None
    
    # 5. Vector search
    if not rag_engine.vectorstore:
        raise ValueError("Vectorstore is empty or not loaded.")
        
    vector_results = []
    initial_k = getattr(config, "VECTOR_K", 25)
    try:
        vector_results = rag_engine.vectorstore.similarity_search(
            query=rewritten_question,
            k=initial_k,
            filter=qdrant_filter
        )
    except Exception as e:
        print(f"[!] Lỗi khi truy vấn vectorstore: {e}")
        
    # 5a. Fallback vector search
    if not vector_results and qdrant_filter:
        try:
            if source_filter:
                relaxed_filter = Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source_filter))])
                vector_results = rag_engine.vectorstore.similarity_search(query=rewritten_question, k=initial_k, filter=relaxed_filter)
            if not vector_results:
                vector_results = rag_engine.vectorstore.similarity_search(query=rewritten_question, k=initial_k)
        except Exception:
            pass

    # 5b. BM25 keyword search
    bm25_results = []
    if hasattr(rag_engine, "bm25_store") and rag_engine.bm25_store:
        bm25_k = getattr(config, "BM25_K", 25)
        bm25_results = rag_engine.bm25_store.search(rewritten_question, k=bm25_k)

    # 5c. Reciprocal Rank Fusion (RRF)
    from source.Generate.generate import reciprocal_rank_fusion
    rrf_k = getattr(config, "RRF_K", 60)
    results = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=rrf_k)
            
    # 6. Cohere Reranking
    results = cohere_rerank(
        query=rewritten_question,
        documents=results,
        cohere_api_key=config.COHERE_API_KEY,
        top_n=config.RERANK_TOP_N
    )
    
    # 7. Generate answer
    qa_chain = get_qa_chain(rag_engine)
    answer = qa_chain.invoke({
        "input": rewritten_question,
        "chat_history": [],
        "context": results
    })
    
    # 8. Retrieve raw contexts
    MAX_CONTEXT_LENGTH = 1200
    contexts = [doc.page_content[:MAX_CONTEXT_LENGTH] for doc in results]
    return answer, contexts

def main():
    print("="*60)
    print("BẮT ĐẦU ĐÁNH GIÁ RAG PIPELINE BẰNG RAGAS")
    print("="*60)
    
    # Load dataset (đường dẫn tuyệt đối từ ROOT_DIR)
    dataset_path = os.path.join(ROOT_DIR, "data", "eval_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"[!] Không tìm thấy file bộ dữ liệu tại: {dataset_path}")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    print(f"[*] Đã tải thành công {len(eval_data)} câu hỏi kiểm thử.")
    
    # RAG LLM: llama-3.1-8b-instant (nhỏ, nhanh cho pipeline)
    from langchain_groq import ChatGroq
    rag_llm = ChatGroq(
       model="llama-3.1-8b-instant",
       temperature=0,
       max_tokens=config.LLM_MAX_TOKENS
    )
    # Eval LLM: llama-3.3-70b-versatile (chất lượng cao hơn để chấm điểm)
    # Dùng Groq thay Gemini để tránh free-tier quota (20 req/ngày)
    eval_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=config.LLM_MAX_TOKENS
    )

    # Init RAG Engine
    print("[*] Đang khởi tạo RAG Engine...")
    rag_engine = FinancialRAG()
    rag_engine.llm = rag_llm  # Ghi đè LLM của RAG để chạy truy vấn và Entity Extraction
    if not rag_engine.load_existing_db():
        print("[!] Không tìm thấy Vectorstore hiện tại.")
        print("[!] Hãy chắc chắn rằng bạn đã chạy scripts/ingest_laws.py hoặc Qdrant đang lưu trữ các tài liệu.")
        return
        
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    
    print("[*] Chạy các câu hỏi qua RAG pipeline để thu thập câu trả lời và ngữ cảnh...")
    for idx, item in enumerate(eval_data):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n  [{idx+1}/{len(eval_data)}] Câu hỏi: {q}")
        try:
            ans, ctxs = query_rag_pipeline(rag_engine, q)
            questions.append(q)
            answers.append(ans)
            contexts_list.append(ctxs)
            ground_truths.append(gt)
            print(f"    -> Đã lấy câu trả lời ({len(ans)} ký tự), Số lượng ngữ cảnh thu hồi: {len(ctxs)}")
        except Exception as e:
            print(f"    [!] Gặp lỗi khi chạy qua pipeline: {e}")
        # Nghỉ 5s giữa các câu hỏi để tránh Groq TPM rate limit
        if idx < len(eval_data) - 1:
            import time; time.sleep(5)
            
    if not questions:
        print("[!] Không có dữ liệu câu trả lời nào được sinh ra để chạy Ragas đánh giá.")
        return
        
    # Prepare HuggingFace Dataset
    print("\n[*] Đang tạo Dataset cho Ragas...")
    dataset_dict = {
        "question": questions,
        "contexts": contexts_list,
        "answer": answers,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    # Setup LLM & Embeddings wraps
    print("[*] Đang wrap mô hình LLM (Groq) và Embeddings (Cohere/HuggingFace) cho Ragas...")
    evaluator_llm = LangchainLLMWrapper(eval_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(rag_engine.embeddings)
    
    print("[*] Đang tiến hành chấm điểm bằng Ragas (Ragas sẽ gọi Groq làm giám khảo)...")
    # Cấu hình: Groq phản hồi nhanh (~1-3s), timeout 60s là đủ dư
    run_cfg = RunConfig(
        timeout=60,         # 60s mỗi lần gọi (Groq rất nhanh)
        max_retries=5,      # thử lại tối đa 5 lần
        max_wait=30,        # chờ tối đa 30s giữa các lần retry
        max_workers=1,      # chạy tuần tự để tránh Groq TPM rate limit
    )
    try:
        result = evaluate(
            dataset=dataset,
            metrics=RAGAS_METRICS,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_cfg,
        )
        
        # Convert to pandas DataFrame
        # pyrefly: ignore [missing-attribute]
        result_df = result.to_pandas()
         
        print("\n" + "="*60)
        print("BẢNG ĐIỂM ĐÁNH GIÁ TRUNG BÌNH (AVERAGE RAGAS SCORES)")
        print("="*60)
        metric_cols = [col for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'] if col in result_df.columns]
        for col in metric_cols:
            avg_score = result_df[col].mean()
            print(f"  - {col:20}: {avg_score:.4f}")
        print("="*60)
        
        # Save detailed logs (đường dẫn tuyệt đối)
        output_csv = os.path.join(ROOT_DIR, "data", "eval_results.csv")
        result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"[*] Đã xuất kết quả chi tiết từng câu hỏi tại: {output_csv}")
        
    except Exception as e:
        print(f"[!] Đã xảy ra lỗi trong quá trình chạy Ragas evaluate: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
