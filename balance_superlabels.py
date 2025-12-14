import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


LABELS = ["Carton", "Glass", "Metal", "Paper", "Plastic", "Other"]


def list_files(root: Path):
    return [p for p in root.iterdir() if p.is_file()]


def copy_file(src: Path, dst: Path, name_counts: dict, dry_run: bool):
    base = src.stem
    ext = src.suffix
    new_name = base + ext
    while (dst / new_name).exists() or name_counts[(dst, new_name)] > 0:
        name_counts[(dst, new_name)] += 1
        new_name = f"{base}__dup{name_counts[(dst, new_name)]}{ext}"
    if not dry_run:
        shutil.copy2(src, dst / new_name)
    return new_name


def balance_split(input_root: Path, output_root: Path, cap: int, seed: int, dry_run: bool):
    rng = random.Random(seed)
    stats = {}
    name_counts = defaultdict(int)

    for label in LABELS:
        src_dir = input_root / label
        dst_dir = output_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = list_files(src_dir) if src_dir.exists() else []
        rng.shuffle(files)

        if len(files) > cap:
            files = files[:cap]

        for f in files:
            copy_file(f, dst_dir, name_counts, dry_run)

        stats[label] = len(files)

    return stats


def copy_split(input_root: Path, output_root: Path, dry_run: bool):
    stats = {}
    name_counts = defaultdict(int)
    for label in LABELS:
        src_dir = input_root / label
        dst_dir = output_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = list_files(src_dir) if src_dir.exists() else []
        for f in files:
            copy_file(f, dst_dir, name_counts, dry_run)
        stats[label] = len(files)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Balance classifier_set_superlabel by capping samples per class."
    )
    parser.add_argument("--input-root", type=Path, default=Path("data/classifier_set_superlabel"))
    parser.add_argument("--output-root", type=Path, default=Path("data/classifier_set_superlabel_balanced"))
    parser.add_argument("--cap", type=int, default=900, help="Max samples per class in balanced splits.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-splits", nargs="+", default=["train"], help="Splits to balance (default: train).")
    parser.add_argument("--copy-splits", nargs="+", default=["val", "test"], help="Splits to copy without balancing.")
    parser.add_argument("--clean-output", action="store_true", help="Remove output_root before writing.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without copying.")
    args = parser.parse_args()

    if args.clean_output and args.output_root.exists() and not args.dry_run:
        shutil.rmtree(args.output_root)

    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = {}

    for split in args.balance_splits:
        src_split = args.input_root / split
        dst_split = args.output_root / split
        if not src_split.exists():
            continue
        stats = balance_split(src_split, dst_split, args.cap, args.seed, args.dry_run)
        summary[split] = stats

    for split in args.copy_splits:
        src_split = args.input_root / split
        dst_split = args.output_root / split
        if not src_split.exists():
            continue
        stats = copy_split(src_split, dst_split, args.dry_run)
        summary[split] = stats

    print("Counts per split/label:")
    for split, stats in summary.items():
        total = sum(stats.values())
        detail = ", ".join(f"{lbl}: {stats.get(lbl, 0)}" for lbl in LABELS)
        print(f"  {split}: total {total} -> {detail}")


if __name__ == "__main__":
    main()
