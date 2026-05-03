try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    import sys
    print("Ultralytics không được tìm thấy. Đang cài đặt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "tensorboard"])
    from ultralytics import YOLO

try:
    import tensorboard
except ImportError:
    import subprocess
    import sys
    print("TensorBoard không được tìm thấy. Đang cài đặt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])

import os
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(os.path.dirname(base_dir), 'Dataset', 'data.yaml')
models_dir = os.path.join(base_dir, "Models")
os.makedirs(models_dir, exist_ok=True)

# ==========================================
# CẤU HÌNH TRAINING CHUNG (GLOBAL CONFIG)
# ==========================================
TRAIN_EPOCHS = 30
TRAIN_IMGSZ = 640
TRAIN_BATCH = 32
PROJECT_DIR_DET = 'runs/detect'
PROJECT_DIR_SEG = 'runs/segment'
# ==========================================

print("="*50)
print("   🚀 Ultralytics YOLO11 Training Script")
print("="*50)

print("\n[1/2] Training Object Detection Model (yolo11n)...")
model_det = YOLO('yolo11n.pt')
# YOLO sẽ tự động chuyển đổi nhãn polygon thành khung chữ nhật (bounding box) để train Detection
results_det = model_det.train(data=yaml_path, epochs=TRAIN_EPOCHS, imgsz=TRAIN_IMGSZ, project=PROJECT_DIR_DET, workers=0, batch=TRAIN_BATCH)

# In ra log độ chính xác
print("\n" + "="*40)
print("📊 DETAILED DETECTION METRICS")
if hasattr(results_det, 'box'):
    print(f" - Độ chính xác (mAP@50): {results_det.box.map50 * 100:.2f}%")
    print(f" - Độ chính xác (mAP@50-95): {results_det.box.map * 100:.2f}%")
if hasattr(model_det, 'trainer') and hasattr(model_det.trainer, 'save_dir'):
    print(f" - Biểu đồ Loss & mAP chi tiết được lưu tại: {model_det.trainer.save_dir}/results.png")
    print(f" - File dữ liệu Loss/Epoch được lưu tại: {model_det.trainer.save_dir}/results.csv")
print("="*40 + "\n")
print("Exporting Detection Model to ONNX...")
det_onnx = model_det.export(format='onnx', opset=12)
if os.path.exists(det_onnx):
    shutil.copy(det_onnx, os.path.join(models_dir, "yolo11n.onnx"))

print("\n[2/2] Training Segmentation Model (yolo11n-seg)...")
model_seg = YOLO('yolo11n-seg.pt')
# Tự động tải coco8-seg dataset mẫu và train
results_seg = model_seg.train(data=yaml_path, epochs=TRAIN_EPOCHS, imgsz=TRAIN_IMGSZ, project=PROJECT_DIR_SEG, workers=0, batch=TRAIN_BATCH)

# In ra log độ chính xác
print("\n" + "="*40)
print("📊 DETAILED SEGMENTATION METRICS")
if hasattr(results_seg, 'seg'):
    print(f" - Độ chính xác bóc tách (Mask mAP@50): {results_seg.seg.map50 * 100:.2f}%")
if hasattr(results_seg, 'box'):
    print(f" - Độ chính xác bắt hộp (Box mAP@50): {results_seg.box.map50 * 100:.2f}%")
if hasattr(model_seg, 'trainer') and hasattr(model_seg.trainer, 'save_dir'):
    print(f" - Biểu đồ Loss & mAP chi tiết được lưu tại: {model_seg.trainer.save_dir}/results.png")
    print(f" - File dữ liệu Loss/Epoch được lưu tại: {model_seg.trainer.save_dir}/results.csv")
print("="*40 + "\n")
print("Exporting Segmentation Model to ONNX...")
seg_onnx = model_seg.export(format='onnx', opset=12)
if os.path.exists(seg_onnx):
    shutil.copy(seg_onnx, os.path.join(models_dir, "yolo11n-seg.onnx"))

print(f"\n[SUCCESS] Models exported to ONNX format in: {os.path.abspath(models_dir)}")
print("="*50)
print("📈 ĐỂ XEM BIỂU ĐỒ TENSORBOARD THEO THỜI GIAN THỰC:")
print("1. Mở một cửa sổ Terminal (Command Prompt) mới.")
print(f"2. Chuyển vào thư mục: cd \"{base_dir}\"")
print("3. Gõ lệnh: tensorboard --logdir runs")
print("4. Mở trình duyệt web và truy cập: http://localhost:6006")
print("="*50)

print("\nVui lòng khởi động lại app 3D-Reconstruction để áp dụng mô hình AI mới!")
print("Nhấn phím Enter để thoát...")
input()
