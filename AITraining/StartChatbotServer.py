import os
import sys
import subprocess

# Ensure necessary packages are installed
def install_packages():
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    
    try:
        import llama_cpp
        import llama_cpp.server
    except ImportError:
        print("Installing llama-cpp-python with server support...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "llama-cpp-python[server]", 
            "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu122"
        ])

install_packages()

from huggingface_hub import hf_hub_download

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(base_dir, "..", "Models"))
os.makedirs(models_dir, exist_ok=True)

# Lấy index model từ tham số dòng lệnh (mặc định là 0)
model_idx = 0
if len(sys.argv) > 1:
    try:
        model_idx = int(sys.argv[1])
    except ValueError:
        model_idx = 0

# Cấu hình danh sách model
MODELS = [
    {"repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF", "filename": "qwen2.5-3b-instruct-q4_k_m.gguf", "desc": "3B (Q4_K_M) ~2.4GB"},
    {"repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF", "filename": "qwen2.5-3b-instruct-q8_0.gguf", "desc": "3B (Q8_0) ~3.5GB"},
    {"repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF", "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "desc": "7B (Q4_K_M) ~4.3GB"},
]

if model_idx < 0 or model_idx >= len(MODELS):
    model_idx = 0

selected_model = MODELS[model_idx]
model_path = os.path.join(models_dir, selected_model["filename"])

print(f"==================================================")
print(f"Da chon model: {selected_model['desc']}")
if not os.path.exists(model_path):
    print(f"Dang tai mo hinh...")
    print(f"Vui long cho (toc do tuy thuoc vao mang cua ban)...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=selected_model["repo_id"], 
            filename=selected_model["filename"], 
            local_dir=models_dir
        )
        print(f"Tai thanh cong! File luu tai: {downloaded_path}")
    except Exception as e:
        print(f"Loi khi tai mo hinh: {e}")
        sys.exit(1)
else:
    print(f"Tim thay mo hinh tai: {model_path}")

def kill_port_owner(port):
    """Tìm và tắt process đang sử dụng port chỉ định (chỉ dành cho Windows)."""
    if os.name != 'nt': return
    try:
        import subprocess
        cmd = f"netstat -ano | findstr :{port}"
        output = subprocess.check_output(cmd, shell=True).decode()
        for line in output.strip().split('\n'):
            if "LISTENING" in line:
                parts = line.strip().split()
                if len(parts) > 0:
                    pid = parts[-1]
                    if pid != "0":
                        print(f"--- Canh bao: Port {port} dang bi chiem boi PID {pid}. Dang giai phong... ---")
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
    except Exception:
        pass

# Giải phóng port 8080 trước khi chạy
kill_port_owner(8080)

print(f"==================================================")
print("\nKhoi chay AI Server tren cong 8080...")
print("Luu y: Cua so nay phai duoc mo de Chatbot hoat dong.")
print("Dang day tinh toan len Card do hoa RTX 3050 (n_gpu_layers=99)...\n")

# Khởi chạy server
server_cmd = [
    sys.executable, "-m", "llama_cpp.server",
    "--model", model_path,
    "--host", "127.0.0.1",
    "--port", "8080",
    "--n_gpu_layers", "99",
    "--n_ctx", "4096"
]

subprocess.run(server_cmd)
