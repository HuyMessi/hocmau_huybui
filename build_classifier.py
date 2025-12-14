# scripts/build_classifier_from_taco_coco.py
"""
Build classifier dataset (6 superlabels) from TACO COCO annotations by cropping bboxes.

Input (TACO):
  data/Taco_dataset/annotations.json
  data/Taco_dataset/batch_*/... (images)

Output:
  data/classifier_set_superlabel/
    train/Carton|Glass|Metal|Paper|Plastic|Other/*.jpg
    val/...
    test/...

Why:
- Classifier train trên ảnh gốc nhiều nền -> học nhầm nền.
- Crop bbox giúp tăng top1_acc + train nhanh hơn.
"""

import argparse
import json
import random
from pathlib import Path
import cv2

# 6 nhãn bạn đã chọn
SUPERLABELS = ["Carton", "Glass", "Metal", "Paper", "Plastic", "Other"]

def to_superlabel(name: str, supercat: str) -> str:
    """
    Map COCO category (name/supercategory) -> 6 superlabels.
    Bạn có thể chỉnh rule nếu dataset bạn khác.
    """
    s = f"{name} {supercat}".lower()

    # carton/cardboard
    if "carton" in s or "cardboard" in s:
        return "Carton"

    # glass
    if "glass" in s:
        return "Glass"

    # metal/aluminum/tin/can
    if "metal" in s or "aluminium" in s or "aluminum" in s or "tin" in s:
        return "Metal"

    # paper
    if "paper" in s:
        return "Paper"

    # plastic
    if "plastic" in s or "styrofoam" in s:
        return "Plastic"

    return "Other"


def safe_imread(img_path: Path):
    img = cv2.imread(str(img_path))
    return img


def crop_bbox(img, x, y, w, h, pad=0.12):
    """
    Crop bbox with padding, keep inside image.
    pad=0.12 nghĩa là nới bbox thêm 12% mỗi chiều.
    """
    H, W = img.shape[:2]
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)

    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad)
    py = int(bh * pad)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(W - 1, x2 + px)
    y2 = min(H - 1, y2 + py)

    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def letterbox_square(img, out_size=224):
    """
    Resize không méo: giữ tỉ lệ, pad thành vuông out_size x out_size.
    """
    h, w = img.shape[:2]
    scale = out_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = 255 * (0 * resized + 1)  # tạo mảng cùng dtype/shape? (sai)
    # cách an toàn:
    import numpy as np
    canvas = np.zeros((out_size, out_size, 3), dtype=resized.dtype)

    # đặt ảnh vào giữa
    top = (out_size - nh) // 2
    left = (out_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas


def split_image_ids(image_ids, seed=42, ratios=(0.8, 0.1, 0.1)):
    random.Random(seed).shuffle(image_ids)
    n = len(image_ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train:n_train+n_val])
    test_ids = set(image_ids[n_train+n_val:])
    return train_ids, val_ids, test_ids
    random.Random(seed).shuffle(image_ids)
    n = len(image_ids)
    n_train = int(n * r_train)
    n_val = int(n * r_val)
    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train:n_train+n_val])
    test_ids = set(image_ids[n_train+n_val:])
    return train_ids, val_ids, test_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taco_root", type=str, default="data/Taco_dataset")
    ap.add_argument("--ann", type=str, default="data/Taco_dataset/annotations.json")
    ap.add_argument("--out", type=str, default="data/classifier_set_superlabel")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_box", type=int, default=32, help="lọc bbox nhỏ (min(w,h) >= min_box)")
    args = ap.parse_args()

    taco_root = Path(args.taco_root)
    ann_path = Path(args.ann)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # tạo folders
    for split in ["train", "val", "test"]:
        for c in SUPERLABELS:
            (out_root / split / c).mkdir(parents=True, exist_ok=True)

    data = json.loads(ann_path.read_text(encoding="utf-8"))

    # maps
    imgid2file = {im["id"]: im["file_name"] for im in data["images"]}
    catid2name = {c["id"]: c.get("name", "") for c in data["categories"]}
    catid2super = {c["id"]: c.get("supercategory", "") for c in data["categories"]}

    image_ids = list(imgid2file.keys())
    train_ids, val_ids, test_ids = split_image_ids(image_ids, seed=args.seed)

    def which_split(img_id: int) -> str:
        if img_id in train_ids:
            return "train"
        if img_id in val_ids:
            return "val"
        return "test"

    # đếm thống kê
    counts = {s: {c: 0 for c in SUPERLABELS} for s in ["train", "val", "test"]}
    missing_imgs = 0
    skipped_small = 0
    saved = 0

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        file_name = imgid2file.get(img_id)
        if not file_name:
            continue

        img_path = taco_root / file_name  # thường file_name đã chứa batch_x/xxx.jpg
        if not img_path.exists():
            missing_imgs += 1
            continue

        img = safe_imread(img_path)
        if img is None:
            continue

        x, y, w, h = ann["bbox"]
        if min(w, h) < args.min_box:
            skipped_small += 1
            continue

        cat_id = ann["category_id"]
        name = catid2name.get(cat_id, "")
        supercat = catid2super.get(cat_id, "")
        sup = to_superlabel(name, supercat)

        crop = crop_bbox(img, x, y, w, h, pad=0.12)
        if crop is None:
            continue

        crop = letterbox_square(crop, out_size=args.imgsz)

        split = which_split(img_id)
        out_name = f"{Path(file_name).stem}_ann{ann['id']}.jpg"
        out_path = out_root / split / sup / out_name
        cv2.imwrite(str(out_path), crop)

        counts[split][sup] += 1
        saved += 1

    print("=== DONE ===")
    print("Saved crops:", saved)
    print("Missing images:", missing_imgs)
    print("Skipped small bbox:", skipped_small)
    print("Counts:")
    for split in ["train", "val", "test"]:
        print(split, counts[split])


if __name__ == "__main__":
    main()
