"""
run_paper.py
------------
Entry point to reproduce all results end-to-end:
  1. Data cleaning & panel construction
  2. Language distribution charts
  3. Programming language trend charts
  4. ChatGPT global availability map
  5. Stata DID / SC / SDID estimations (base model)
  6. Merge control variables + Stata estimations with controls
  7. Patch \\label{} into generated Stata table files
  8. Compile Tesis.tex → Tesis.pdf (two pdflatex passes)

Requirements:
  pip install -r requirements.txt
  Stata 15+ installed and accessible from the system path
  (or set STATA_EXE inside code/all_code.py)
  pdflatex installed (MiKTeX or TeX Live)
"""

import subprocess
import sys
import os
import re
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Step 1-6: run the full analysis pipeline ──────────────────────────────────

print("=" * 60)
print("STEP 1-6  Running full replication pipeline: code/all_code.py")
print("=" * 60)

result = subprocess.run(
    [sys.executable, os.path.join(BASE_DIR, "code", "all_code.py")],
    cwd=BASE_DIR,
)

if result.returncode != 0:
    print(f"\nPipeline exited with error code {result.returncode}. Aborting.")
    sys.exit(result.returncode)


# ── Step 7: patch \\label{} into Stata-generated table files ─────────────────
# all_code.py overwrites these files each run, so we re-inject the labels here.

print("\n" + "=" * 60)
print("STEP 7    Patching \\label{} into Stata table files")
print("=" * 60)

TABLE_LABELS = {
    os.path.join(BASE_DIR, "output", "tables", "gpt_impact_github_DataScience.tex"):
        "tab:tabla3",
    os.path.join(BASE_DIR, "output", "tables", "gpt_impact_github_DataScience_controls.tex"):
        "tab:tabla5",
}

for fpath, label in TABLE_LABELS.items():
    if not os.path.isfile(fpath):
        print(f"  WARNING: {fpath} not found, skipping label patch.")
        continue

    with open(fpath, encoding="utf-8") as f:
        txt = f.read()

    if f"\\label{{{label}}}" in txt:
        print(f"  {os.path.basename(fpath)}: label already present, skipping.")
        continue

    # Insert \label immediately after the first \caption{...}
    patched = re.sub(
        r'(\\caption\{[^}]+\})',
        rf'\1\n\\label{{{label}}}',
        txt,
        count=1,
    )

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"  {os.path.basename(fpath)}: \\label{{{label}}} added.")


# ── Step 8: compile Tesis.tex → Tesis.pdf (two passes) ───────────────────────

print("\n" + "=" * 60)
print("STEP 8    Compiling Tesis.tex with pdflatex (2 passes)")
print("=" * 60)

def find_pdflatex():
    """Locate pdflatex on Windows (MiKTeX / TeX Live) or Unix."""
    import platform, shutil

    # Check PATH first
    found = shutil.which("pdflatex")
    if found:
        return found

    if platform.system() == "Windows":
        candidates = []
        # MiKTeX user install
        appdata = os.environ.get("LOCALAPPDATA", "")
        if appdata:
            candidates.append(
                os.path.join(appdata, "Programs", "MiKTeX", "miktex", "bin", "x64", "pdflatex.exe")
            )
        # MiKTeX system install
        for pf in [r"C:\Program Files", r"C:\Program Files (x86)"]:
            candidates.append(os.path.join(pf, "MiKTeX", "miktex", "bin", "x64", "pdflatex.exe"))
        # TeX Live
        for year in range(2025, 2019, -1):
            candidates.append(rf"C:\texlive\{year}\bin\windows\pdflatex.exe")
            candidates.append(rf"C:\texlive\{year}\bin\win32\pdflatex.exe")
        for path in candidates:
            if os.path.isfile(path):
                return path

    return "pdflatex"   # fallback: hope it is in PATH


pdflatex = find_pdflatex()
tex_file  = os.path.join(BASE_DIR, "Tesis.tex")

for pass_num in (1, 2):
    print(f"\n  pdflatex pass {pass_num}/2 ...")
    r = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", tex_file],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    # Show only relevant lines
    for line in r.stdout.splitlines():
        if any(kw in line for kw in ("Error", "error", "Output written", "pages")):
            if not any(skip in line for skip in ("Font", "microtype", "babel", "natbib", "ignored")):
                print("   ", line)
    if r.returncode != 0:
        print(f"\n  pdflatex exited with error code {r.returncode}.")
        last_lines = deque((r.stdout or "").splitlines(), maxlen=80)
        print("\n".join(last_lines))
        sys.exit(r.returncode)

print("\n" + "=" * 60)
print("DONE  Tesis.pdf is up to date.")
print("=" * 60)
