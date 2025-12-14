"""
GIAI ĐOẠN 4 - Train YOLO Detector (1 class: trash-object) phù hợp Intel iGPU (train CPU)

Yêu cầu cài:
  pip install ultralytics

Input:
  configs/detector_data.yaml   (trỏ tới data/detector_set với images/labels train/val/test)

Output:
  models/detector/runs/<RUN_NAME>/weights/best.pt
  models/detector/best/trash_object_best.pt  (copy từ best.pt để bạn dùng chạy webcam sau này)
"""

from pathlib import Path
import shutil
import os

from ultralytics import YOLO


# =========================
# CẤU HÌNH CHUNG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_YAML = PROJECT_ROOT / "configs" / "detector_data.yaml"

RUNS_DIR = PROJECT_ROOT / "models" / "detector" / "runs"
BEST_DIR = PROJECT_ROOT / "models" / "detector" / "best"

# Model khuyến nghị cho CPU: YOLO11n (nhẹ) để iteration nhanh
# (Bạn có thể đổi sang "yolo11s.pt" nếu muốn accuracy hơn nhưng sẽ chậm hơn trên CPU)
MODEL_WEIGHTS = r"E:\DHKT\Nam3\Ki1\Machine Learning\Garbage Classification Project\Garbage Classification Code\models\detector\runs\taco_trash_yolo11n_cpu_v1\weights\last.pt"

# ===== Tham số train (gợi ý cho CPU) =====
EPOCHS = 200          # CPU train chậm -> 80 là mức hợp lý; muốn tốt hơn có thể 120+
RESUME = False
IMGSZ = 640          # 640 thường cho accuracy tốt; nếu máy yếu -> 512
BATCH = 8            # CPU: 4–16 tuỳ RAM/CPU. Nếu bị chậm/đơ -> hạ xuống 4
WORKERS = 0          # Windows + CPU: 0 ổn định nhất. Nếu muốn nhanh hơn thử 2–4
DEVICE = "cpu"       # Intel Iris Xe: train bằng CPU (Ultralytics train GPU chủ yếu cho CUDA)

# Early stopping: nếu không cải thiện sau N epochs thì dừng để tiết kiệm thời gian
PATIENCE = 20

# Đặt tên run để không ghi đè (v1, v2, v3...)
RUN_NAME = "taco_trash_yolo11n_cpu_v2"

# Cache ảnh vào RAM để tăng tốc? (dataset lớn có thể tốn RAM -> mặc định False)
CACHE = False


def sanity_check_paths():
    """Kiểm tra file cấu hình dataset có tồn tại trước khi train."""
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Không thấy file dataset yaml: {DATA_YAML}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)


def train():
    """
    Train detector.
    Lưu ý:
    - Dataset của bạn đã là 1 class (trash-object), nhưng vẫn set single_cls=True để an toàn.
    """
    print("=== Stage 4: Training YOLO detector ===")
    print(f"Data yaml : {DATA_YAML}")
    print(f"Model     : {MODEL_WEIGHTS}")
    print(f"Device    : {DEVICE}")
    print(f"Epochs    : {EPOCHS} | imgsz={IMGSZ} | batch={BATCH} | workers={WORKERS}")

    model = YOLO(MODEL_WEIGHTS)

    # results = model.train(
    #     data=str(DATA_YAML),
    #     epochs=EPOCHS,
    #     imgsz=IMGSZ,
    #     batch=BATCH,
    #     workers=WORKERS,
    #     device=DEVICE,

    #     # quality & stability
    #     patience=PATIENCE,
    #     single_cls=True,   # vì detector chỉ có 1 class
    #     cache=CACHE,

    #     # quản lý output gọn trong dự án
    #     project=str(RUNS_DIR),
    #     name=RUN_NAME,
    #     exist_ok=False,    # tránh vô tình ghi đè run cũ
    #     plots=True         # xuất biểu đồ loss/metrics
    # )
    #Code mới
    # ====== RESUME LOGIC (quan trọng) ======
    run_dir = RUNS_DIR / RUN_NAME
    last_ckpt = run_dir / "weights" / "last.pt"   # đường dẫn checkpoint để resume

    if RESUME:
        if not last_ckpt.exists():
            raise FileNotFoundError(
                f"RESUME=True nhưng không thấy checkpoint: {last_ckpt}\n"
                f"Hãy kiểm tra folder run có đúng tên không: {run_dir}"
            )

        print(f"Resuming from: {last_ckpt}")
        model = YOLO(str(last_ckpt))  # nạp last.pt để tiếp tục
        results = model.train(
            resume=True,              # tiếp tục training từ checkpoint
            epochs=EPOCHS,            # tổng epoch bạn muốn đạt (vd 250)
            device=DEVICE,
            workers=WORKERS,
            imgsz=IMGSZ,
            batch=BATCH
        )

    else:
        # Train mới từ pretrained
        model = YOLO(MODEL_WEIGHTS)
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            workers=WORKERS,
            device=DEVICE,

            patience=PATIENCE,
            single_cls=True,
            cache=CACHE,

            project=str(RUNS_DIR),
            name=RUN_NAME,
            exist_ok=False,
            plots=True
        )

    # Ultralytics trả về results có save_dir (thư mục của run)
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (RUNS_DIR / RUN_NAME)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    print(f"\nRun folder : {save_dir}")
    print(f"Best weight: {best_pt}")
    print(f"Last weight: {last_pt}")

    if not best_pt.exists():
        raise FileNotFoundError(f"Không thấy best.pt sau khi train: {best_pt}")

    # Copy best.pt sang models/detector/best/ để inference sau này
    final_best = BEST_DIR / "trash_object_best.pt"
    shutil.copy2(best_pt, final_best)
    print(f"\n Copied best weight to: {final_best}")

    return final_best


def quick_eval(best_weight: Path):
    """
    (Tuỳ chọn) Evaluate nhanh.
    - Mặc định Ultralytics val trên split=val
    - Nếu bạn muốn test set, thử split='test' (tuỳ version Ultralytics hỗ trợ).
    """
    print("\n=== Quick evaluation ===")
    model = YOLO(str(best_weight))

    # Val on val split
    _ = model.val(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        batch=1,          # eval batch nhỏ cho CPU
        device=DEVICE
    )
    print("Done validation on 'val' split.")

    # Nếu Ultralytics của bạn hỗ trợ split='test' thì bật thêm (nếu lỗi thì comment lại)
    try:
        _ = model.val(
            data=str(DATA_YAML),
            imgsz=IMGSZ,
            batch=1,
            device=DEVICE,
            split="test"
        )
        print("Done validation on 'test' split.")
    except TypeError:
        print("Version Ultralytics của bạn chưa hỗ trợ split='test' trong model.val(). Bỏ qua test split.")


if __name__ == "__main__":
    sanity_check_paths()
    best = train()
    quick_eval(best)