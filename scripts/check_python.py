"""
Run the self-check of every lesson's Python / PyTorch file (src/**/*.py).

Each file ends with `if __name__ == "__main__": _self_check()`, which asserts that the
code does what the lesson says on small synthetic data (no downloads).

    pip install -r requirements.txt
    python scripts/check_python.py            # all files
    python scripts/check_python.py vit unet   # only files whose path contains these words
"""
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    filters = sys.argv[1:]
    files = sorted(p for p in (ROOT / "src").rglob("*.py") if not filters or any(f in str(p) for f in filters))
    failed = []
    for path in files:
        start = time.time()
        result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=900)
        rel = path.relative_to(ROOT)
        took = f"{time.time() - start:5.1f}s"
        if result.returncode == 0:
            print(f"✓ {took}  {rel}  {result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ''}")
        else:
            failed.append(rel)
            print(f"✗ {took}  {rel}\n{result.stdout}{result.stderr}")
    print(f"\n{len(files) - len(failed)}/{len(files)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
