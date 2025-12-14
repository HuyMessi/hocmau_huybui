import os
import json
import random
import shutil
from pathlib import Path


# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

TACO_DIR = PROJECT_ROOT / "data" / "Taco_dataset"
COCO_JSON = TACO_DIR / "annotations.json"

OUT_DIR = PROJECT_ROOT / "data" / "detector_set"
OUT_IMAGES = OUT_DIR / "images"
OUT_LABELS = OUT_DIR / "labels"

# Tỉ lệ chia tập
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

# Chỉ 1 lớp duy nhất
CLASS_NAME = "trash_object"
CLASS_ID = 0


# =========================
# HÀM TIỆN ÍCH
# =========================
def ensure_dirs():
    """Tạo đủ thư mục output YOLO: images/labels cho train/val/test."""
    for split in ["train", "val", "test"]:
        (OUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
        (OUT_LABELS / split).mkdir(parents=True, exist_ok=True)


def index_taco_images(taco_dir: Path):
    """
    Tạo index để tìm ảnh nhanh trong các batch_*.
    Trả về:
      - by_rel: map "batch_1/xxx.jpg" -> full_path
      - by_name: map "xxx.jpg" -> full_path (trường hợp COCO file_name không có batch)
    """
    by_rel = {}
    by_name = {}

    # Quét tất cả batch_*
    for p in taco_dir.glob("batch_*"):
        if not p.is_dir():
            continue
        for img in p.rglob("*"):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            # key dạng batch_1/xxx.jpg (path tương đối từ taco_dir)
            rel = img.relative_to(taco_dir).as_posix()
            by_rel[rel] = img

            # key dạng xxx.jpg (basename)
            # NOTE: Nếu trùng tên ở nhiều batch thì by_name sẽ bị ghi đè.
            # TACO thường ít trùng, nhưng vẫn có thể xảy ra.
            by_name[img.name] = img

    return by_rel, by_name


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    COCO bbox = [x, y, w, h] (pixel, x-y là góc trên-trái)
    YOLO bbox = [x_center, y_center, w, h] chuẩn hoá 0..1
    """
    x, y, w, h = bbox

    # Clip để tránh bbox vượt biên (an toàn)
    x = max(0.0, float(x))
    y = max(0.0, float(y))
    w = max(0.0, float(w))
    h = max(0.0, float(h))

    if w <= 1 or h <= 1:
        return None  # bbox quá nhỏ / lỗi

    # đảm bảo không vượt kích thước ảnh
    if x >= img_w or y >= img_h:
        return None
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    x_center = x + w / 2.0
    y_center = y + h / 2.0

    # Chuẩn hoá
    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h

    # Clip lần cuối về [0,1]
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    return x_center, y_center, w, h


def split_list(items, train_ratio, val_ratio, test_ratio, seed=42):
    """Chia list theo tỉ lệ."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def write_yolo_label(label_path: Path, yolo_lines):
    """Ghi file nhãn YOLO (.txt). Nếu không có line nào, vẫn tạo file rỗng (tuỳ bạn)."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for line in yolo_lines:
            f.write(line + "\n")


def safe_copy(src: Path, dst: Path):
    """Copy ảnh sang folder mới, giữ metadata."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    # 1) Check input
    if not COCO_JSON.exists():
        raise FileNotFoundError(f"Khong tim thay: {COCO_JSON}")

    ensure_dirs()

    # 2) Index ảnh trong các batch_*
    print("[1/5] Indexing TACO images...")
    by_rel, by_name = index_taco_images(TACO_DIR)
    print(f"  Found images: {len(by_rel)}")

    # 3) Load COCO json
    print("[2/5] Loading COCO annotations...")
    with open(COCO_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    # Map image_id -> (file_path, width, height, file_name)
    img_map = {}
    missing = 0

    for img in images:
        image_id = img["id"]
        file_name = img["file_name"]
        width = int(img.get("width", 0))
        height = int(img.get("height", 0))

        # Tìm đường dẫn ảnh thật
        full_path = None

        # TH1: file_name có dạng batch_x/xxx.jpg
        if file_name in by_rel:
            full_path = by_rel[file_name]
        else:
            # TH2: file_name chỉ là xxx.jpg
            base = Path(file_name).name
            if base in by_name:
                full_path = by_name[base]

        if full_path is None or not full_path.exists():
            missing += 1
            continue

        # width/height đôi khi thiếu trong COCO -> lấy từ OpenCV/PIL sẽ cần code khác
        # Ở đây TACO thường có sẵn width/height. Nếu 0, ta vẫn cố dùng, nhưng bbox->yolo sẽ fail.
        img_map[image_id] = {
            "path": full_path,
            "file_name": full_path.name,  # dùng basename để đặt tên trong detector_set
            "width": width,
            "height": height,
        }

    if missing > 0:
        print(f"  WARNING: {missing} images in COCO json could not be matched to files.")

    print(f"  Matched images usable: {len(img_map)}")

    # 4) Gom annotations theo image_id (gộp nhãn về 1 lớp trash-object)
    print("[3/5] Converting COCO bbox -> YOLO (single class)...")
    ann_by_img = {}
    bad_bbox = 0

    for ann in annotations:
        image_id = ann["image_id"]
        if image_id not in img_map:
            continue

        bbox = ann.get("bbox", None)
        if bbox is None:
            # Nếu dataset chỉ có segmentation mà không có bbox, bạn sẽ cần bước chuyển segmentation->bbox
            continue

        img_w = img_map[image_id]["width"]
        img_h = img_map[image_id]["height"]
        if img_w <= 0 or img_h <= 0:
            bad_bbox += 1
            continue

        yolo = coco_bbox_to_yolo(bbox, img_w, img_h)
        if yolo is None:
            bad_bbox += 1
            continue

        x_center, y_center, w, h = yolo
        # YOLO line: class_id x y w h
        line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

        ann_by_img.setdefault(image_id, []).append(line)

    if bad_bbox > 0:
        print(f"  NOTE: Skipped bad bboxes: {bad_bbox}")

    # 5) Split train/val/test theo image list
    print("[4/5] Splitting dataset train/val/test...")
    image_ids = list(img_map.keys())
    train_ids, val_ids, test_ids = split_list(image_ids, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, seed=RANDOM_SEED)
    print(f"  train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    split_map = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    # 6) Copy ảnh + ghi nhãn
    print("[5/5] Writing YOLO files (copy images + labels)...")
    total_written = 0

    for split, ids in split_map.items():
        for image_id in ids:
            info = img_map[image_id]
            src_img = info["path"]
            base_name = Path(info["file_name"]).stem  # abc.jpg -> abc

            dst_img = OUT_IMAGES / split / src_img.name
            dst_label = OUT_LABELS / split / f"{base_name}.txt"

            # Copy ảnh
            safe_copy(src_img, dst_img)

            # Ghi label (có thể rỗng nếu ảnh không có annotation)
            yolo_lines = ann_by_img.get(image_id, [])
            write_yolo_label(dst_label, yolo_lines)

            total_written += 1

    print("\nDONE.")
    print(f"- Wrote images: {total_written}")
    print(f"- Output folder: {OUT_DIR}")
    print(f"- Single class: id={CLASS_ID}, name='{CLASS_NAME}'")

    # (Tuỳ chọn) tạo yaml config cho YOLO (Ultralytics) trong configs/
    configs_dir = PROJECT_ROOT / "configs"
    configs_dir.mkdir(exist_ok=True)
    yaml_path = configs_dir / "detector_data.yaml"
    yaml_text = f"""# YOLO dataset config
path: {OUT_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    print(f"- Wrote YOLO data config: {yaml_path}")


if __name__ == "__main__":
    main()
