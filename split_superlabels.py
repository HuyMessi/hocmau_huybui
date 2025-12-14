import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

LABELS = ["Carton", "Glass", "Metal", "Paper", "Plastic", "Other"]

# Map 60 TACO categories -> 5 superlabels + Other
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
    """Index images by relative path and basename for quick lookup."""
    by_rel = {}
    by_name = {}
    for p in taco_dir.glob("batch_*"):
        if not p.is_dir():
            continue
        for img in p.rglob("*"):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rel = img.relative_to(taco_dir).as_posix()
            by_rel[rel] = img
            by_name[img.name] = img
    return by_rel, by_name


def compute_image_labels(coco: dict) -> dict:
    """Return dict image_id -> set(labels)."""
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


def split_items(items, train_ratio, val_ratio, seed):
    # test ratio inferred as the remainder
    assert 0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0
    assert train_ratio + val_ratio < 1.0
    rnd = random.Random(seed)
    items = list(items)
    rnd.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def ensure_dirs(output_root: Path, splits, dry_run: bool):
    if dry_run:
        return
    for split in splits:
        for lbl in LABELS:
            (output_root / split / lbl).mkdir(parents=True, exist_ok=True)


def copy_to_split(files, label, split, output_root: Path, taco_dir: Path, dry_run: bool):
    for src in files:
        # use batch_xx__filename to avoid name collisions across batches
        try:
            rel = src.relative_to(taco_dir)
            dest_name = rel.as_posix().replace("/", "__")
        except ValueError:
            dest_name = src.name
        dest = output_root / split / label / dest_name
        if dry_run:
            continue
        if not dest.exists():
            shutil.copy2(src, dest)


def main():
    root = Path(__file__).resolve().parent.parent
    default_taco = root / "data" / "Taco_dataset"
    default_ann = default_taco / "annotations.json"
    default_out = root / "data" / "classifier_set_superlabel"

    parser = argparse.ArgumentParser(
        description="Split TACO into 5+1 superlabel train/val/test (80/10/10 by default)"
    )
    parser.add_argument("--taco-dir", type=Path, default=default_taco)
    parser.add_argument("--annotations", type=Path, default=default_ann)
    parser.add_argument("--output-root", type=Path, default=default_out)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing train/val/test under output_root before writing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report counts, do not copy files",
    )
    args = parser.parse_args()

    coco = load_annotations(args.annotations)
    by_rel, by_name = build_indices(args.taco_dir)
    image_labels = compute_image_labels(coco)

    if args.clean_output and not args.dry_run:
        for split in ["train", "val", "test"]:
            split_dir = args.output_root / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    ensure_dirs(args.output_root, ["train", "val", "test"], args.dry_run)

    # Gather files per label
    files_per_label = defaultdict(set)
    missing_files = []
    unlabeled = 0

    for img in coco["images"]:
        rel = img["file_name"]
        src = by_rel.get(rel) or by_name.get(Path(rel).name)
        if not src:
            missing_files.append(rel)
            continue
        labels = image_labels.get(img["id"])
        if not labels:
            labels = {"Other"}
            unlabeled += 1
        for lbl in labels:
            files_per_label[lbl].add(src)

    # Split and copy
    stats = defaultdict(lambda: defaultdict(int))
    for lbl in LABELS:
        files = list(files_per_label.get(lbl, []))
        train_files, val_files, test_files = split_items(
            files, args.train_ratio, args.val_ratio, seed=args.seed
        )
        for split_name, split_files in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]:
            copy_to_split(
                split_files,
                lbl,
                split_name,
                args.output_root,
                args.taco_dir,
                args.dry_run,
            )
            stats[split_name][lbl] = len(split_files)

    print("Counts per split/label:")
    for split in ["train", "val", "test"]:
        total = sum(stats[split].values())
        label_counts = ", ".join(f"{lbl}: {stats[split][lbl]}" for lbl in LABELS)
        print(f"  {split}: total {total} -> {label_counts}")

    if missing_files:
        print(f"Missing files: {len(missing_files)} (showing first 5): {missing_files[:5]}")
    if unlabeled:
        print(f"Images without annotations defaulted to Other: {unlabeled}")


if __name__ == "__main__":
    main()
