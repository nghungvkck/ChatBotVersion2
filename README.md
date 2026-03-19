# PDF RAG Assistant (ChatBotVersion2)

Đây là một ứng dụng **RAG (Retrieval-Augmented Generation)** dùng **Streamlit** để chat với dữ liệu từ file PDF. Ứng dụng dùng:

- **LangChain + Chroma** (vector DB)
- **HuggingFace Transformers** (LLM + embedding)
- **LangChain Hub** để tải prompt RAG

## 🖼️ Minh hoạ (images)

1. `images/image.png`
2. `images/image-1.png`
3. `images/xy_ly_text.jpg`

![Overview](./images/image.png)

![Vector DB](./images/image-1.png)

![Xử lý text](./images/xy_ly_text.jpg)

---

## 🚀 Bắt đầu (Cài đặt môi trường)

### 1) Chuẩn bị Python

- Khuyến nghị dùng **Python 3.11+**.
- Tạo và kích hoạt virtual environment (đảm bảo không cài thư viện toàn cục):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> Nếu dùng cmd: `.
.venv\Scripts\activate.bat`

### 2) Cài đặt dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> Nếu bạn gặp lỗi do thiếu `streamlit`, cài thêm:
> `pip install streamlit`

---

## ▶️ Chạy ứng dụng

1) Chạy Streamlit app:

```powershell
streamlit run app.py
```

2) Trên trình duyệt, truy cập `http://localhost:8501` (nếu không tự mở).

3) Upload file PDF, nhấn **Xử lý PDF**, rồi đặt câu hỏi.

---

## 📂 Cấu trúc chính của project (giải thích mục đích từng file)

- `app.py` - **Giao diện Streamlit + điều phối luồng xử lý**: tải model, upload PDF, gọi pipeline RAG, hiển thị kết quả.
- `core/config.py` - **Cấu hình chung** (model name, max token, v.v.) để dễ thay đổi model/param.
- `core/llm_loader.py` - **Tải và tạo LLM pipeline** (TinyLlama, quantization 4-bit, HuggingFace pipeline).
- `core/embedding_loader.py` - **Tải embedding model** dùng để chuyển văn bản thành vector.
- `rag/pdf_processor.py` - **Xử lý PDF & tách văn bản**: đọc PDF, chia thành các chunk có ngữ nghĩa.
- `rag/rag_chain.py` - **Xây dựng RAG pipeline**: tạo vector database (Chroma), retriever, prompt, và gọi LLM để trả lời.
- `requirements.txt` - **Danh sách thư viện cần cài** để chạy ứng dụng.
- `test.py` - **Ví dụ demo khác**; dùng trực tiếp LangChain + Transformers để test ý tưởng (không dùng `core/`, `rag/`).

---

## 📝 Chạy thử nghiệm (tùy chọn)

- `test.py` là một phiên bản demo khác dùng trực tiếp LangChain + Transformers.

---

## 💡 Ghi chú

- Model mặc định tải từ HuggingFace (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) và có thể lớn, cần GPU hoặc cấu hình đủ mạnh.
- Nếu dùng CPU hoặc bị thiếu RAM, cân nhắc dùng model nhỏ hơn hoặc bật `device_map="auto"` và `load_in_4bit=True` (đã cấu hình sẵn trong `core/llm_loader.py`).

