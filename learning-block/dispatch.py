# learning-block/dispatch.py
import os, sys, runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
selector = os.path.join(ROOT, "active_model.txt")
models_dir = os.path.join(ROOT, "models")

default_file = "mobilenetv2_96_baseline.py"

try:
    selected = open(selector, "r").read().strip()
    if not selected:
        selected = default_file
except FileNotFoundError:
    selected = default_file

# allow "mobilenetv2_96_baseline" or "... .py"
if not selected.endswith(".py"):
    selected += ".py"

target = os.path.join(models_dir, selected)

if not os.path.isfile(target):
    print(f"[dispatcher] ERROR: '{target}' not found.\nAvailable models:")
    try:
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".py"):
                print(" -", f)
    except Exception:
        pass
    sys.exit(1)

print(f"[dispatcher] Launching: {os.path.relpath(target, ROOT)}")
# Ensure argv[0] looks like the launched script, and pass through EI args unchanged.
sys.argv[0] = target
runpy.run_path(target, run_name="__main__")
