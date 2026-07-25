import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

from source.Function.search_Qdrant import FinancialRAG
from source.Database.db_connection import SQLDatabase

# Initialize Database (eager — lightweight, chỉ kết nối SQL)
db = SQLDatabase()
rag_engine: Optional[FinancialRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler: khởi tạo RAG engine 1 lần duy nhất khi server start,
    thay vì khởi tạo lần đầu khi có request (tránh làm chậm trang chủ).
    """
    global rag_engine
    print("[*] Khởi tạo RAG Engine khi server start...")
    rag_engine = FinancialRAG()
    rag_engine.load_existing_db()
    print("[*] RAG Engine sẵn sàng.")
    yield
    # Shutdown: cleanup nếu cần
    print("[*] Server đang tắt.")


app = FastAPI(title="Trợ lý Luật pháp Việt Nam AI", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def get_rag_engine() -> FinancialRAG:
    """Trả về RAG engine đã được khởi tạo sẵn từ lifespan."""
    if rag_engine is None:
        raise RuntimeError("RAG engine chưa được khởi tạo.")
    return rag_engine


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[int] = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the homepage."""
    # rag_engine đã sẵn sàng từ lifespan — không còn khởi tạo ở đây nữa
    has_vs = rag_engine is not None and rag_engine.vectorstore is not None
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={
            "use_db": True, 
            "db_error_msg": None,
            "has_vectorstore": has_vs
        }
    )


@app.post("/start-session")
async def start_session():
    """Create a new chat session."""
    return {"session_id": db.create_session()}


@app.get("/get-sessions")
async def get_sessions():
    """Get all sessions to display in the sidebar."""
    sessions = []
    for s in db.get_all_sessions():
        sessions.append({
            "id": s["id"],
            "pdf_name": s["pdf_name"],
            "title": s["title"],
            "created_at": s["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        })
    return {"sessions": sessions}


@app.get("/get-chat-history/{session_id}")
async def get_chat_history(session_id: int):
    """Retrieve chat history for a session."""
    history = db.get_chat_history_with_sources(session_id)
    summary_info = db.get_session_summary(session_id)
    return {
        "chat_history": history, 
        "pdf_summary": summary_info["pdf_summary"] if summary_info else None
    }


@app.delete("/delete-session/{session_id}")
async def delete_session(session_id: int):
    """Delete a chat session."""
    db.delete_session(session_id)
    return {"status": "success"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), session_id: Optional[int] = Form(None)):
    """Upload a PDF, save it, run indexing, and generate a summary."""
    filename = file.filename
    if not filename or not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    laws_dir = os.path.join("data", "laws")
    os.makedirs(laws_dir, exist_ok=True)
    save_path = os.path.join(laws_dir, filename)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    if session_id is None:
        session_id = db.create_session(filename, "")
        
    num_chunks = get_rag_engine().load_and_index_pdf(save_path, session_id)
    if num_chunks == 0:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=400, detail="File PDF scan không chứa văn bản số.")
        
    summary = get_rag_engine().summarize_pdf(save_path)
    db.update_session_pdf(session_id, filename, summary)
    
    return {"status": "success", "session_id": session_id, "summary": summary}


@app.post("/ask")
async def ask_chatbot(request: QueryRequest):
    """Ask a question, running RAG query and saving dialog turn."""
    query = request.query
    session_id = request.session_id
    
    chat_history = db.get_chat_history(session_id) if session_id else []
    
    is_summary_query = any(kw in query.lower() for kw in ["tóm tắt", "tom tat", "summary"])
    summary_info = db.get_session_summary(session_id) if session_id else None
    summary_text = summary_info["pdf_summary"] if summary_info else None
    
    if is_summary_query and summary_text:
        answer, sources = summary_text, []
    else:
        if not get_rag_engine().vectorstore:
            raise HTTPException(status_code=400, detail="Thư viện luật hiện tại đang trống.")
        answer, sources = get_rag_engine().ask(query, chat_history, session_id)
        
    if session_id is None:
        session_id = db.create_session()
        
    db.save_message(session_id, "user", query)
    db.save_message(session_id, "assistant", answer, sources=sources)
    
    return {"answer": answer, "sources": sources, "session_id": session_id}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
