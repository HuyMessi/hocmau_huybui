import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["Carton", "Glass", "Metal", "Paper", "Plastic", "Other"]

CATEGORY_TO_LABEL = {
    "Aluminum foil": "Metal",
    "Battery": "Metal",
    "Aluminum blister pack": "Metal",
    "Carded blister pack": "Plastic",
    "Other plastic bottle": "Plastic",
    "Clear plastic bottle": "Plastic",
    "Glass bottle": "Glass",
    "Plastic bottle cap": "Plastic",
    "Metal bottle cap": "Metal",
    "Broken glass": "Glass",
    "Food Can": "Metal",
    "Aerosol Can": "Metal",
    "Drink can": "Metal",
    "Toilet tube": "Carton",
    "Other carton": "Carton",
    "Egg carton": "Carton",
    "Drink carton": "Carton",
    "Corrugated carton": "Carton",
    "Meal carton": "Carton",
    "Pizza box": "Carton",
    "Paper cup": "Paper",
    "Disposable plastic cup": "Plastic",
    "Foam cup": "Plastic",
    "Glass cup": "Glass",
    "Other plastic cup": "Plastic",
    "Food waste": "Other",
    "Glass jar": "Glass",
    "Plastic lid": "Plastic",
    "Metal lid": "Metal",
    "Other plastic": "Plastic",
    "Magazine paper": "Paper",
    "Tissues": "Paper",
    "Wrapping paper": "Paper",
    "Normal paper": "Paper",
    "Paper bag": "Paper",
    "Plastified paper bag": "Paper",
    "Plastic film": "Plastic",
    "Six pack rings": "Plastic",
    "Garbage bag": "Plastic",
    "Other plastic wrapper": "Plastic",
    "Single-use carrier bag": "Plastic",
    "Polypropylene bag": "Plastic",
    "Crisp packet": "Plastic",
    "Spread tub": "Plastic",
    "Tupperware": "Plastic",
    "Disposable food container": "Plastic",
    "Foam food container": "Plastic",
    "Other plastic container": "Plastic",
    "Plastic gloves": "Plastic",
    "Plastic utensils": "Plastic",
    "Pop tab": "Metal",
    "Rope & strings": "Other",
    "Scrap metal": "Metal",
    "Shoe": "Other",
    "Squeezable tube": "Plastic",
    "Plastic straw": "Plastic",
    "Paper straw": "Paper",
    "Styrofoam piece": "Plastic",
    "Unlabeled litter": "Other",
    "Cigarette": "Other",
}


def load_annotations(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_indices(taco_dir: Path):
    index_by_rel = {}
    index_by_name = {}
    for p in taco_dir.glob("batch_*"):
        if not p.is_dir():
            continue
        for img in p.rglob("*"):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rel = img.relative_to(taco_dir).as_posix()
            index_by_rel[rel] = img
            index_by_name[img.name] = img
    return index_by_rel, index_by_name


def compute_image_labels(coco: dict):
    catid_to_label = {
        c["id"]: CATEGORY_TO_LABEL.get(c["name"], "Other")
        for c in coco["categories"]
    }
    image_labels = defaultdict(set)
    for ann in coco["annotations"]:
        image_labels[ann["image_id"]].add(
            catid_to_label.get(ann["category_id"], "Other")
        )
    return image_labels


def ensure_output_dirs(output_dir: Path, dry_run: bool):
    if dry_run:
        return
    for lbl in LABELS:
        (output_dir / lbl).mkdir(parents=True, exist_ok=True)


def copy_images(coco: dict, taco_dir: Path, output_dir: Path, dry_run: bool = False):
    index_by_rel, index_by_name = build_indices(taco_dir)
    image_labels = compute_image_labels(coco)

    ensure_output_dirs(output_dir, dry_run)

    stats = Counter()
    missing_files = []
    unlabeled = 0
    multilabel = 0

    for img in coco["images"]:
        rel = img["file_name"]
        img_path = index_by_rel.get(rel) or index_by_name.get(Path(rel).name)
        if not img_path:
            missing_files.append(rel)
            continue

        labels = image_labels.get(img["id"])
        if not labels:
            labels = {"Other"}
            unlabeled += 1

        if len(labels) > 1:
            multilabel += 1

        for lbl in labels:
            dest = output_dir / lbl / img_path.name
            if not dry_run and not dest.exists():
                shutil.copy2(img_path, dest)
            stats[lbl] += 1

    return stats, missing_files, unlabeled, multilabel


def parse_args():
    root = Path(__file__).resolve().parent.parent
    default_taco = root / "data" / "Taco_dataset"
    default_ann = default_taco / "annotations.json"
    default_out = root / "data" / "classifier_set_superlabel" / "train"

    parser = argparse.ArgumentParser(
        description="Build 5+1 superlabel folders from TACO annotations"
    )
    parser.add_argument("--taco-dir", type=Path, default=default_taco)
    parser.add_argument("--annotations", type=Path, default=default_ann)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--dry-run", action="store_true", help="Only report counts")
    return parser.parse_args()


def main():
    args = parse_args()
    coco = load_annotations(args.annotations)
    stats, missing_files, unlabeled, multilabel = copy_images(
        coco, args.taco_dir, args.output_dir, dry_run=args.dry_run
    )

    print("Copied counts per label:")
    for lbl in LABELS:
        print(f"  {lbl}: {stats[lbl]}")
    print(f"Images with multiple labels (duplicated across folders): {multilabel}")
    if missing_files:
        print(f"Missing files: {len(missing_files)} (first 5 shown): {missing_files[:5]}")
    if unlabeled:
        print(f"Images without annotations defaulted to Other: {unlabeled}")


if __name__ == "__main__":
    main()
