import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

import config

class FinancialRAG:
    def __init__(self):
        # Initialize Embeddings theo provider
        if config.EMBEDDING_PROVIDER == "cohere":
            from langchain_cohere import CohereEmbeddings
          
            self.embeddings = CohereEmbeddings(
                cohere_api_key=config.COHERE_API_KEY,
                model=config.COHERE_EMBEDDING_MODEL,
            )
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL_NAME
            )
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
        
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=config.QDRANT_PATH)
        
        # Will hold the Qdrant vector store
        self.vectorstore = None

    def load_and_index_pdf(self, file_path, session_id=None):
        """Loads a PDF, splits it, and stores it in Qdrant Local."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        # Gắn nhãn session_id vào metadata của mỗi chunk để lọc sau này
        if session_id is not None:
            for doc in splits:
                doc.metadata["session_id"] = session_id

        # Kiểm tra và khởi tạo collection nếu chưa có
        try:
            self.client.get_collection(config.COLLECTION_NAME)
        except Exception:
            from qdrant_client.models import VectorParams, Distance
            vector_size = len(self.embeddings.embed_query("test"))
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

        # Nếu có dùng SQL Server, chỉ xóa các vector cũ của đúng session này trước khi nạp mới
        if session_id is not None:
            try:
                
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                self.client.delete(
                    collection_name=config.COLLECTION_NAME,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.session_id",
                                match=MatchValue(value=session_id)
                            )
                        ]
                    )
                )
            except Exception:
                pass
        else:
            # Fallback (chế độ in-memory, không DB): xóa toàn bộ collection
            try:
                self.client.delete_collection(config.COLLECTION_NAME)
            except Exception:
                pass
                
            from qdrant_client.models import VectorParams, Distance
            vector_size = len(self.embeddings.embed_query("test"))
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=config.COLLECTION_NAME,
            embedding=self.embeddings
        )
        self.vectorstore.add_documents(splits)
        return len(splits)

    def load_existing_db(self):
        """Loads an existing Qdrant DB if it exists."""
        if os.path.exists(config.QDRANT_PATH):
            # Check if collection exists
            try:
                self.client.get_collection(config.COLLECTION_NAME)
                self.vectorstore = QdrantVectorStore(
                    client=self.client, 
                    collection_name=config.COLLECTION_NAME, 
                    embedding=self.embeddings
                )
                return True
            except Exception:
                pass
        return False

    def get_conversation_chain(self, session_id=None):
        """Builds a history-aware RAG chain."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Please load a PDF first.")

        search_kwargs = {"k": config.RETRIEVER_K}
        
        # Nếu có session_id, thực hiện lọc metadata của Qdrant chỉ lấy các vector thuộc session hiện tại
        if session_id is not None:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            search_kwargs["filter"] = Filter(
                must=[
                    FieldCondition(
                        key="metadata.session_id",
                        match=MatchValue(value=session_id)
                    )
                ]
            )

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        # 1. Contextualize Question Prompt (deals with history)
        contextualize_q_system_prompt = """Bạn là trợ lý AI. Dựa vào lịch sử hội thoại và câu hỏi mới nhất của người dùng,
        hãy viết lại câu hỏi thành một câu hỏi độc lập có đầy đủ ý nghĩa. Không cần trả lời câu hỏi, chỉ cần viết lại nếu cần thiết, ngược lại giữ nguyên."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # 2. Answer Question Prompt
        qa_system_prompt = """Bạn là một chuyên gia phân tích tài chính cao cấp (Financial Analyst).
        Nhiệm vụ của bạn là phân tích báo cáo tài chính và trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp dưới đây.
        
        Quy tắc:
        - Chỉ sử dụng dữ liệu trong Context để trả lời. Không bịa đặt số liệu.
        - Nếu số liệu hoặc thông tin không có trong Context, hãy nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu."
        - Trả lời rõ ràng, mạch lạc, có thể dùng bullet points để dễ đọc. Trích dẫn nếu cần.
        - Nếu câu hỏi yêu cầu so sánh hoặc tính toán đơn giản từ các số liệu có sẵn, hãy thực hiện cẩn thận.
        
        Context:
        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # 3. Full Retrieval Chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain

    def ask(self, question, chat_history, session_id=None):
        """Asks a question with history and returns the answer and sources."""
        chain = self.get_conversation_chain(session_id)
        # chat_history format should be a list of BaseMessage (HumanMessage, AIMessage)
        response = chain.invoke({"input": question, "chat_history": chat_history})
        
        answer = response["answer"]
        sources = []
        for doc in response.get("context", []):
            sources.append(f"Page {doc.metadata.get('page', 'Unknown')}")
            
        # Deduplicate sources
        sources = list(set(sources))
        return answer, sources

    def summarize_pdf(self, file_path):
        """Extracts text from PDF and gets a full summary from LLM using stuffing."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Để tránh vượt giới hạn Rate Limit (12,000 Tokens/phút) của Groq:
        # Nếu file dài hơn 10 trang, chúng ta chọn lọc lấy 8 trang đầu (chứa tổng quan/số liệu chính)
        # và 2 trang cuối (chứa kết luận kiểm toán/chữ ký) để tóm tắt.
        num_pages = len(docs)
        if num_pages <= 10:
            selected_docs = docs
        else:
            selected_docs = docs[:8] + docs[-2:]
            
        # Combine selected page contents
        full_text = "\n".join([doc.page_content for doc in selected_docs])
        
        # Cắt tiếp nếu tổng số ký tự vẫn quá lớn (giới hạn an toàn khoảng 28,000 ký tự ~ 7,000 tokens)
        max_chars = 28000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n\n[Tài liệu đã được lược bớt một số trang giữa để tránh vượt giới hạn API]"
            
        prompt = f"""Bạn là một chuyên gia phân tích tài chính cao cấp (Financial Analyst).
Nhiệm vụ của bạn là đọc toàn bộ văn bản báo cáo tài chính/bản cáo bạch/tài liệu dưới đây và viết một bản tóm tắt phân tích tài chính toàn diện, rõ ràng bằng tiếng Việt.

Hãy tự động nhận diện loại hình tổ chức (ví dụ: Ngân hàng, Doanh nghiệp sản xuất/thương mại/dịch vụ, Công ty chứng khoán, Công ty bảo hiểm, Tập đoàn đa ngành,...) và điều chỉnh nội dung tóm tắt cho phù hợp nhất với đặc thù ngành nghề đó. Hãy tập trung xoay quanh đúng thông tin có trong file báo cáo, không áp đặt một kịch bản cố định nếu tài liệu không phù hợp.

Yêu cầu nội dung bản tóm tắt cần làm rõ:
1. **Thông tin chung về tổ chức:** 
   - Xác định rõ tên đầy đủ của tổ chức (Ngân hàng, Công ty, Tổng công ty, Tập đoàn,...) và loại hình hoạt động chính.
   - Thời kỳ/Niên độ báo cáo (ví dụ: Quý 1/2024, Năm 2023,...).
2. **Kết quả hoạt động kinh doanh:**
   - Trình bày các chỉ số doanh thu/thu nhập và lợi nhuận cốt lõi phù hợp với ngành. 
   - *Ví dụ đối với Doanh nghiệp thông thường:* Doanh thu thuần, Giá vốn, Lợi nhuận gộp, Lợi nhuận trước & sau thuế, các chỉ số tăng trưởng.
   - *Ví dụ đối với Ngân hàng:* Thu nhập lãi thuần, Thu nhập ngoài lãi, Chi phí hoạt động, Chi phí dự phòng rủi ro tín dụng, Lợi nhuận trước & sau thuế.
   - *Ví dụ đối với Công ty chứng khoán:* Doanh thu hoạt động (tự doanh, môi giới, cho vay ký quỹ), Chi phí hoạt động, Lợi nhuận.
3. **Tình hình tài chính & Cấu trúc vốn:**
   - Trình bày các khoản mục tài sản và nguồn vốn trọng yếu phù hợp nhất với loại hình hoạt động.
   - *Ví dụ đối với Doanh nghiệp thông thường:* Tiền và tương đương tiền, Phải thu khách hàng, Hàng tồn kho, Tài sản cố định, Nợ phải trả (đặc biệt là nợ vay ngắn/dài hạn), Vốn chủ sở hữu.
   - *Ví dụ đối với Ngân hàng:* Tổng tài sản, Cho vay khách hàng, Dự phòng rủi ro cho vay, Tiền gửi của khách hàng, Phát hành giấy tờ có giá, Tỷ lệ nợ xấu (NPL) nếu có, Vốn chủ sở hữu.
4. **Điểm nhấn nổi bật & Rủi ro:**
   - Các điểm đáng chú ý trong kỳ báo cáo (tăng trưởng đột biến, các biến động lớn về tài sản/nguồn vốn, dòng tiền, hoặc các dự án lớn).
   - Rủi ro nổi bật (rủi ro thanh khoản, rủi ro nợ xấu, biến động lãi suất, tỷ giá, thị trường,...).
5. **Ý kiến kiểm toán (nếu có):**
   - Ý kiến của đơn vị kiểm toán độc lập (chấp nhận toàn phần, ngoại trừ, trái ngược, từ chối đưa ra ý kiến).

Văn bản tài liệu:
---
{full_text}
---

Hãy viết bản tóm tắt một cách tự nhiên, mạch lạc, trực tiếp đi vào các số liệu của tổ chức trong tài liệu, sử dụng đúng thuật ngữ chuyên ngành tài chính của loại hình tổ chức đó.

BẢN TÓM TẮT BÁO CÁO TÀI CHÍNH CHI TIẾT:"""

        response = self.llm.invoke(prompt)
        return response.content
