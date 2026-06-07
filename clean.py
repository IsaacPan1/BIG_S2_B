"""clean.py — return the repo to the clean state we run from.

Removes everything `python main.py` produces, leaving only inputs and source:

  reports/                — wiped except .gitkeep
  data/features_*.parquet — fold-bound feature blocks (regenerated per run)
  submission.csv, report.pdf, report.txt — repo-root run outputs
  __pycache__/            — bytecode caches

Preserves: data/* inputs, tools/ source, main.py, metadata files.

Usage:
    python clean.py            # clean
    python clean.py --dry-run  # show what would be removed without deleting
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _iter_targets() -> list[Path]:
    """Enumerate every file/dir the cleaner would remove."""
    targets: list[Path] = []

    # 1) reports/* (preserve .gitkeep)
    reports = ROOT / "reports"
    if reports.exists():
        for entry in reports.iterdir():
            if entry.name == ".gitkeep":
                continue
            targets.append(entry)

    # 2) Fold-bound feature parquets under data/
    data = ROOT / "data"
    if data.exists():
        targets.extend(data.glob("features_train_fold_*.parquet"))
        targets.extend(data.glob("features_valid_fold_*.parquet"))
        # Legacy global feature parquets from the old architecture, if present
        for legacy in ("features_train.parquet", "features_val.parquet"):
            p = data / legacy
            if p.exists():
                targets.append(p)

    # 3) Repo-root run outputs
    for name in ("submission.csv", "report.pdf", "report.txt"):
        p = ROOT / name
        if p.exists():
            targets.append(p)

    # 4) __pycache__ caches anywhere in the tree
    targets.extend(ROOT.rglob("__pycache__"))

    return targets


def _remove(p: Path) -> None:
    if p.is_dir():
        # On Windows the currently-running interpreter may hold an open handle
        # on its own __pycache__. Best-effort removal: ignore PermissionErrors
        # for cache dirs (they'll be cleaned on the next non-running invocation).
        if p.name == "__pycache__":
            shutil.rmtree(p, ignore_errors=True)
            if p.exists():
                raise PermissionError(
                    f"{p} is locked (likely held by the running interpreter); skipped"
                )
        else:
            shutil.rmtree(p)
    else:
        p.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description="Reset the repo to a clean state.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be removed; remove nothing.")
    args = ap.parse_args()

    targets = sorted(set(_iter_targets()))
    if not targets:
        print("Already clean — nothing to remove.")
        return 0

    verb = "Would remove" if args.dry_run else "Removing"
    print(f"{verb} {len(targets)} item(s):")
    n_removed = 0
    for p in targets:
        rel = p.relative_to(ROOT).as_posix()
        kind = "dir " if p.is_dir() else "file"
        print(f"  [{kind}] {rel}")
        if not args.dry_run:
            try:
                _remove(p)
                n_removed += 1
            except PermissionError as e:
                print(f"    skipped: {e}")
            except Exception as e:
                print(f"    !! failed: {e}")

    # Ensure reports/.gitkeep still exists after wiping reports/
    if not args.dry_run:
        keep = ROOT / "reports" / ".gitkeep"
        keep.parent.mkdir(parents=True, exist_ok=True)
        if not keep.exists():
            keep.touch()
        print()
        print(f"Removed {n_removed} item(s). Repo is clean.")
        print("Run `python main.py` to regenerate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
