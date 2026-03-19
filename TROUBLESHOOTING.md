# 🚨 Tổng hợp lỗi thường gặp & cách sửa (ChatBotVersion2)

Dưới đây là tổng hợp các lỗi phổ biến gặp phải khi cài đặt và chạy dự án **PDF RAG Assistant**, cùng nguyên nhân và cách khắc phục.

---

## 1. ❌ Lỗi: "Execution Policy" khi kích hoạt venv (PowerShell)

### 🔍 Nguyên nhân
PowerShell chặn chạy script `.ps1` do chính sách bảo mật (Execution Policy) trên máy.

### ✅ Cách sửa
Mở PowerShell (không cần admin) và chạy:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

Sau đó kích hoạt venv:
```powershell
.\.venv\Scripts\Activate.ps1
```

**Nếu không muốn thay đổi policy:** Dùng `activate.bat` (chạy trong cmd hoặc PowerShell):
```powershell
.\.venv\Scripts\activate.bat
```

---

## 2. ❌ Lỗi: "No matching distribution found for torch==2.2.2"

### 🔍 Nguyên nhân
- Bạn đang dùng **Python 3.13.x**.
- PyTorch 2.2.x **chưa hỗ trợ Python 3.13** (chỉ có wheel cho 3.11/3.12).

### ✅ Cách sửa
#### Giải pháp 1 (khuyến nghị): Dùng Python 3.11/3.12
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Giải pháp 2: Cập nhật torch trong requirements.txt
Đã sửa sẵn: `torch==2.2.2` → `torch==2.10.0` (tương thích Python 3.13).

---

## 3. ❌ Lỗi: "Cargo, the Rust package manager, is not installed"

### 🔍 Nguyên nhân
Một số package (như `tokenizers`, `sentence-transformers`) cần **Rust toolchain** để build native extensions, nhưng máy chưa cài Rust/Cargo.

### ✅ Cách sửa
1. Cài Rust toolchain: Vào **https://rustup.rs** và follow hướng dẫn (thường là chạy script và restart terminal).
2. Kiểm tra cài đặt:
```powershell
rustc --version
cargo --version
```
3. Chạy lại:
```powershell
pip install -r requirements.txt
```

**Nếu không muốn cài Rust:** Cập nhật pip/setuptools và thử lại:
```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 4. ❌ Lỗi: "Conflicting dependencies" (transformers vs langchain-huggingface)

### 🔍 Nguyên nhân
- `transformers 4.37.2` yêu cầu `tokenizers<0.19 and >=0.14`
- `langchain-huggingface 0.2.0` yêu cầu `tokenizers>=0.19.1`
- Xung đột vì phiên bản cũ không tương thích.

### ✅ Cách sửa
Đã cập nhật `requirements.txt`:
- `transformers==4.37.2` → `transformers>=4.40.0`
- `accelerate==0.27.2` → `accelerate>=0.30.0`

Phiên bản mới của transformers hỗ trợ `tokenizers>=0.19.1`.

---

## 💡 Lưu ý chung

- **Luôn kích hoạt venv trước khi cài package:** `.\.venv\Scripts\Activate.ps1`
- **Nếu gặp lỗi pip version:** Chạy `python -m pip install --upgrade pip`
- **Nếu model tải chậm/lỗi:** Kiểm tra kết nối internet hoặc dùng VPN.
- **Nếu thiếu RAM/GPU:** Thay model nhỏ hơn trong `core/config.py` (vd: TinyLlama → một model khác).

---

## 📞 Nếu vẫn lỗi

- Kiểm tra Python version: `python --version`
- Kiểm tra venv: `which python` (nên là trong `.venv`)
- Cài lại venv: Xóa thư mục `.venv` và tạo lại.
- Hoặc chia sẻ lỗi cụ thể để hỗ trợ thêm.

---

*Tổng hợp từ các lần cài đặt dự án ChatBotVersion2. Cập nhật lần cuối: 2026-03-19*</content>
<parameter name="filePath">d:\ChatBotVersion2\TROUBLESHOOTING.md