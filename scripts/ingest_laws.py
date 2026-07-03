import os
import sys
import glob
import time
from dotenv import load_dotenv

# Thêm thư mục gốc (project root) vào sys.path để import config và source
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# Load env variables first
load_dotenv(os.path.join(ROOT_DIR, ".env"))

import config

def main():
    laws_dir = os.path.join(ROOT_DIR, "data", "laws")
    os.makedirs(laws_dir, exist_ok=True)
    
    # Lấy danh sách file PDF trong thư mục data/laws
    pdf_files = glob.glob(os.path.join(laws_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"[*] Thư mục '{laws_dir}' trống hoặc không tìm thấy file PDF nào.")
        print("[*] Vui lòng đặt các file PDF luật vào thư mục này rồi chạy lại script, hoặc tải lên trực tiếp từ giao diện Web.")
        return
        
    print(f"[*] Tìm thấy {len(pdf_files)} file PDF luật để nạp...")
    
    # Khởi tạo RAG Engine
    from source.Function.search_Qdrant import FinancialRAG
    rag_engine = FinancialRAG()
    
    for idx, pdf_path in enumerate(pdf_files):
        file_name = os.path.basename(pdf_path)
        print(f"\n[+] Đang xử lý ({idx+1}/{len(pdf_files)}): {file_name}...")
        try:
            num_chunks = rag_engine.load_and_index_pdf(pdf_path)
            print(f"[-] Thành công! Đã nạp và lập chỉ mục {num_chunks} chunks cho: {file_name}")
        except Exception as e:
            print(f"[!] Lỗi khi xử lý {file_name}: {str(e)}")

        # Chờ 65 giây giữa các file để tránh Cohere trial rate limit (100k tokens/phút)
        if idx < len(pdf_files) - 1:
            wait_sec = 65
            print(f"[~] Chờ {wait_sec}s để tránh rate limit Cohere Embedding API...")
            time.sleep(wait_sec)
            
    print("\n[*] Quá trình nạp luật hoàn tất!")

if __name__ == "__main__":
    main()
