import os
import sys
import json

# Ensure parent directory is in path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from source.Function.utils import extract_law_name_from_filename

def extract_entities_from_query(rag_engine, query: str):
    """Dùng LLM để trích xuất Điều, Chương và tên Luật từ câu hỏi của người dùng."""
    prompt = f"""Bạn là trợ lý AI chuyên về luật pháp Việt Nam. Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và trích xuất các thông tin cấu trúc sau (nếu có):
    1. Số Điều luật (Ví dụ: "Điều 5", "Điều 100", "Điều 10a")
    2. Số Chương (Ví dụ: "Chương I", "Chương III")
    3. Loại văn bản hoặc Tên luật (Ví dụ: "Luật Đất Đai", "Luật Hôn nhân", "Nghị định 15")
    
    Hãy trả về kết quả dưới định dạng JSON với các khóa: "article", "chapter", "law_name". Nếu không có thông tin nào, hãy để giá trị là null.
    Không giải thích gì thêm, chỉ trả về chuỗi JSON hợp lệ.
    
    Ví dụ:
    "Điều kiện cấp sổ đỏ theo Điều 100 Luật Đất Đai là gì?" -> {{"article": "Điều 100", "chapter": null, "law_name": "Luật Đất Đai"}}
    "Quy định tại Chương II Luật Hôn nhân" -> {{"article": null, "chapter": "Chương II", "law_name": "Luật Hôn nhân"}}
    "Độ tuổi đăng ký kết hôn là bao nhiêu?" -> {{"article": null, "chapter": null, "law_name": null}}
    
    Câu hỏi: "{query}"
    JSON:"""
    
    try:
        response = rag_engine.llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content.strip())
        return data
    except Exception:
        return {"article": None, "chapter": None, "law_name": None}

def rewrite_question(rag_engine, question, chat_history):
    """Viết lại câu hỏi dựa trên lịch sử chat để làm câu hỏi độc lập."""
    if not chat_history:
        return question
        
    prompt = f"""Bạn là trợ lý AI chuyên về luật pháp. Dựa vào lịch sử hội thoại và câu hỏi mới nhất của người dùng,
    hãy viết lại câu hỏi thành một câu hỏi độc lập có đầy đủ ý nghĩa pháp lý để phục vụ việc tra cứu. Không cần giải thích hay trả lời câu hỏi, chỉ cần viết lại nếu cần thiết, ngược lại giữ nguyên câu hỏi.
    
    Lịch sử chat:
    {chat_history}
    
    Câu hỏi mới: {question}
    Câu hỏi viết lại:"""
    
    try:
        response = rag_engine.llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return question

def get_qa_chain(rag_engine):
    """Tạo chain trả lời câu hỏi dựa trên Context."""
    qa_system_prompt = """Bạn là một chuyên gia tư vấn luật pháp Việt Nam chuyên nghiệp (Legal Advisor).
    Nhiệm vụ của bạn là giải đáp thắc mắc của người dùng dựa trên thông tin ngữ cảnh (Context) các văn bản luật được cung cấp dưới đây.
    
    Quy tắc:
    - Chỉ sử dụng dữ liệu trong Context để trả lời. Không tự bịa đặt điều luật, số hiệu văn bản pháp lý hoặc thông tin không có trong tài liệu.
    - Trích dẫn rõ ràng tên văn bản luật, số hiệu, điều, khoản, điểm (Ví dụ: "Theo Điều 5 Luật Đất đai 2024...") khi trả lời để câu trả lời có tính thuyết phục cao.
    - Linh hoạt về thuật ngữ: Nếu người dùng sử dụng từ ngữ đời thường hoặc thuật ngữ không chuẩn xác tuyệt đối nhưng có ý nghĩa tương đương hoặc liên quan mật thiết với nội dung trong Context (ví dụ: dùng "vô ý giết người" thay vì "vô ý làm chết người"), hãy chủ động giải đáp dựa trên các điều luật liên quan đó và hướng dẫn lại thuật ngữ pháp lý chính xác cho họ. Không trả lời cứng nhắc là "không tìm thấy" trong trường hợp này.
    - Chỉ trả lời: "Tôi không tìm thấy thông tin pháp lý này trong cơ sở dữ liệu luật hiện tại của hệ thống." khi nội dung câu hỏi hoàn toàn lệch hướng, không liên quan hoặc Context không chứa bất kỳ thông tin nào liên quan.
    - Trả lời rõ ràng, mạch lạc bằng tiếng Việt, đúng thuật ngữ pháp lý. Có thể dùng các gạch đầu dòng để phân tách các quy định pháp luật cho người dùng dễ đọc.
    
    Context:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    return create_stuff_documents_chain(rag_engine.llm, qa_prompt)

def cohere_rerank(query: str, documents: list, cohere_api_key: str, top_n: int = 7):
    """Tái xếp hạng danh sách tài liệu sử dụng Cohere Rerank API qua urllib."""
    if not cohere_api_key or not documents:
        return documents[:top_n]
        
    import urllib.request
    import urllib.error
    import json
    
    url = "https://api.cohere.com/v1/rerank"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"bearer {cohere_api_key}"
    }
    
    doc_texts = [doc.page_content for doc in documents]
    
    payload = {
        "model": getattr(config, "RERANK_MODEL", "rerank-multilingual-v3.0"),
        "query": query,
        "documents": doc_texts,
        "top_n": top_n
    }
    
    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers=headers, 
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            
        reranked_results = []
        for result in res_data.get("results", []):
            idx = result["index"]
            if idx < len(documents):
                documents[idx].metadata["rerank_score"] = result["relevance_score"]
                reranked_results.append(documents[idx])
        return reranked_results
    except Exception as e:
        print(f"[!] Lỗi khi gọi Cohere Rerank API: {e}. Fallback sử dụng thứ tự từ vectorstore.")
        return documents[:top_n]

def reciprocal_rank_fusion(
    vector_results: list,
    bm25_results: list,
    rrf_k: int = 60
) -> list:
    """
    Reciprocal Rank Fusion (Cormack et al. 2009) — kết hợp 2 danh sách kết quả
    từ vector search và BM25 search thành một danh sách được sắp xếp theo RRF score.

    Công thức: RRF(d) = Σ 1/(k + rank_i(d)) với k=60 (thường dùng theo paper gốc)

    Args:
        vector_results: Documents từ Qdrant similarity_search (thứ tự = rank)
        bm25_results:   Documents từ BM25 search (thứ tự = rank)
        rrf_k:          Hằng số RRF (mặc định 60)

    Returns:
        List[Document] sắp xếp theo RRF score giảm dần (deduped theo page_content).
    """
    from langchain_core.documents import Document

    scores: dict[str, float] = {}   # key = page_content hash, value = RRF score
    doc_map: dict[str, Document] = {}  # key = page_content hash, value = Document

    for rank, doc in enumerate(vector_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        if key not in doc_map:
            doc_map[key] = doc
            doc_map[key].metadata["rrf_sources"] = []
        doc_map[key].metadata.setdefault("rrf_sources", []).append("vector")

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        if key not in doc_map:
            doc_map[key] = doc
            doc_map[key].metadata["rrf_sources"] = []
        doc_map[key].metadata.setdefault("rrf_sources", []).append("bm25")

    # Sắp xếp theo RRF score giảm dần
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    fused = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = round(scores[key], 6)
        fused.append(doc)

    return fused


def ask(rag_engine, question, chat_history, session_id=None):
    """Asks a question with history and returns the answer and sources."""
    # 1. Viết lại câu hỏi độc lập
    rewritten_question = rewrite_question(rag_engine, question, chat_history)
    
    # 2. Trích xuất thực thể từ câu hỏi viết lại
    entities = extract_entities_from_query(rag_engine, rewritten_question)
    
    # 3. Ánh xạ law_name sang tên file nguồn trong Qdrant
    source_filter = None
    if entities.get("law_name"):
        normalized_law_name = entities["law_name"].lower().replace(" ", "").replace("đ", "d")
        indexed_docs = rag_engine.get_indexed_documents()
        for doc_name in indexed_docs:
            # So sánh với tên file thô
            normalized_doc = doc_name.lower().replace(" ", "")
            # So sánh với law_name đã được làm sạch từ tên file (xử lý dấu gạch, viết hoa)
            cleaned_law_name = extract_law_name_from_filename(doc_name).lower().replace(" ", "")
            if (normalized_law_name in normalized_doc
                    or normalized_doc in normalized_law_name
                    or normalized_law_name in cleaned_law_name
                    or cleaned_law_name in normalized_law_name):
                source_filter = doc_name
                break
    
    # 4. Tạo bộ lọc Qdrant Filter
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    must_conditions = []
    
    if source_filter:
        must_conditions.append(
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value=source_filter)
            )
        )
        
    if entities.get("article"):
        must_conditions.append(
            FieldCondition(
                key="metadata.article",
                match=MatchValue(value=entities["article"])
            )
        )
        
    if entities.get("chapter"):
        must_conditions.append(
            FieldCondition(
                key="metadata.chapter",
                match=MatchValue(value=entities["chapter"])
            )
        )
        
    qdrant_filter = Filter(must=must_conditions) if must_conditions else None
    
    # 5. Vector search: truy vấn Qdrant (top VECTOR_K ứng viên)
    if not rag_engine.vectorstore:
        raise ValueError("Vectorstore not initialized.")
        
    vector_results = []
    initial_k = config.VECTOR_K
    try:
        vector_results = rag_engine.vectorstore.similarity_search(
            query=rewritten_question,
            k=initial_k,
            filter=qdrant_filter
        )
    except Exception:
        pass
        
    # 5a. Fallback nếu không có kết quả với bộ lọc cứng
    if not vector_results and qdrant_filter:
        try:
            if source_filter:
                relaxed_filter = Filter(must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source_filter)
                    )
                ])
                vector_results = rag_engine.vectorstore.similarity_search(
                    query=rewritten_question,
                    k=initial_k,
                    filter=relaxed_filter
                )
            if not vector_results:
                vector_results = rag_engine.vectorstore.similarity_search(
                    query=rewritten_question,
                    k=initial_k
                )
        except Exception:
            pass

    # 5b. BM25 keyword search (song song với vector search)
    bm25_results = []
    if hasattr(rag_engine, "bm25_store") and rag_engine.bm25_store:
        bm25_k = getattr(config, "BM25_K", 25)
        bm25_results = rag_engine.bm25_store.search(rewritten_question, k=bm25_k)

    # 5c. Reciprocal Rank Fusion — merge vector + BM25 results
    rrf_k = getattr(config, "RRF_K", 60)
    results = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=rrf_k)

    # 6. Cohere Reranking — chọn top_n tài liệu liên quan nhất từ merged results
    results = cohere_rerank(
        query=rewritten_question,
        documents=results,
        cohere_api_key=config.COHERE_API_KEY,
        top_n=config.RERANK_TOP_N
    )
            
    # 7. Trả lời câu hỏi bằng chain
    qa_chain = get_qa_chain(rag_engine)
    response = qa_chain.invoke({
        "input": rewritten_question,
        "chat_history": chat_history,
        "context": results
    })
    
    # 8. Định dạng nguồn dẫn chiếu chi tiết
    sources = []
    for doc in results:
        doc_source = doc.metadata.get('source', 'Tài liệu luật')
        doc_page = doc.metadata.get('page', 0) + 1
        doc_article = doc.metadata.get('article')
        doc_chapter = doc.metadata.get('chapter')
        
        ref_str = f"{doc_source} (Trang {doc_page})"
        if doc_article:
            if doc_chapter and doc_chapter != "Chưa rõ":
                ref_str = f"{doc_article} ({doc_chapter}) - {ref_str}"
            else:
                ref_str = f"{doc_article} - {ref_str}"
        sources.append(ref_str)
        
    sources = list(set(sources))
    return response, sources
