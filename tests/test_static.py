import py_compile
from pathlib import Path
from unittest.mock import patch

p=Path(__file__).resolve().parents[1]
for f in ("fm.py", "probe.py", "tabby_smoke.py"):
    py_compile.compile(str(p/f), doraise=True)
assert (p/"job.example.json").exists()
assert (p/"install.sh").exists()
assert (p/"run.sh").exists()
print("STATIC_OK")
