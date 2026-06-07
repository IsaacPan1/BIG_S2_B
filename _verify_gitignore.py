"""Simulate `git status --ignored` to confirm no generated file is tracked.

This is a stand-in for `git` (not on PATH on this machine). It walks the repo,
applies .gitignore rules with pathspec, and lists:
  - generated files that are correctly ignored
  - any generated file that would be TRACKED (would be a contract violation)
"""
from pathlib import Path
import pathspec

ROOT = Path(__file__).resolve().parent
GITIGNORE = ROOT / ".gitignore"

with open(GITIGNORE) as f:
    spec = pathspec.PathSpec.from_lines("gitwildmatch", f.readlines())

GENERATED_GLOBS = [
    "reports/*",
    "data/features_train_fold_*.parquet",
    "data/features_valid_fold_*.parquet",
    "submission.csv",
    "report.pdf",
    "report.txt",
]

tracked_violations = []
ignored_ok = []

for pattern in GENERATED_GLOBS:
    for p in ROOT.glob(pattern):
        rel = p.relative_to(ROOT).as_posix()
        if rel.endswith("/.gitkeep"):
            continue
        if spec.match_file(rel):
            ignored_ok.append(rel)
        else:
            tracked_violations.append(rel)

print(f"Generated files correctly ignored: {len(ignored_ok)}")
for r in ignored_ok[:6]:
    print(f"  ignored: {r}")
if len(ignored_ok) > 6:
    print(f"  ... and {len(ignored_ok)-6} more")
print()
if tracked_violations:
    print(f"!! Tracked generated files (CONTRACT VIOLATION): {len(tracked_violations)}")
    for r in tracked_violations:
        print(f"  TRACKED: {r}")
    raise SystemExit(1)
else:
    print("OK: every generated file is ignored.")

# Confirm reports/.gitkeep IS tracked (not ignored).
keep = (ROOT / "reports" / ".gitkeep").relative_to(ROOT).as_posix()
if spec.match_file(keep):
    print(f"!! {keep} is ignored — gitkeep negation rule broken")
    raise SystemExit(1)
else:
    print(f"OK: {keep} is preserved (not ignored).")
