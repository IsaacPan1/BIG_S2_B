# MUST remain Python 3.6-compatible — no __future__ imports, no 3.7+ syntax —
# or it cannot report a sub-3.11 interpreter, defeating its purpose.
import sys

v = sys.version_info
if v < (3, 11):
    sys.stderr.write(
        "PREFLIGHT FAILED: Python 3.11 required (3.11.x), detected "
        + str(v.major) + "." + str(v.minor) + "." + str(v.micro)
        + " via `python3.11`.\n"
        "Fix: ensure python3.11 is available (create your venv from Python 3.11):\n"
        "  python3.11 -m venv /path/to/envs/my_env\n"
        "  source /path/to/envs/my_env/bin/activate   # Linux/macOS\n"
        "  python3.11 --version   # must show 3.11.x\n"
        "See README Section 3 for full setup instructions.\n"
    )
    sys.exit(1)

print("PREFLIGHT OK: Python " + str(v.major) + "." + str(v.minor) + "." + str(v.micro))
