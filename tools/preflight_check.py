# MUST remain Python 3.6-compatible — no __future__ imports, no 3.7+ syntax —
# or it cannot report a sub-3.11 interpreter, defeating its purpose.
import sys

v = sys.version_info
if v < (3, 11):
    sys.stderr.write(
        "PREFLIGHT FAILED: Python 3.11+ required, detected "
        + str(v.major) + "." + str(v.minor) + "." + str(v.micro)
        + " via bare `python`.\n"
        "Fix: activate the project venv before launching Claude Code:\n"
        "  source /path/to/envs/award_b/bin/activate   # Linux/macOS\n"
        "  python --version   # must show 3.11+\n"
        "See README Section 3 for full setup instructions.\n"
    )
    sys.exit(1)

print("PREFLIGHT OK: Python " + str(v.major) + "." + str(v.minor) + "." + str(v.micro))
