from pathlib import Path
import sys
import runpy

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))


def run_script(filename: str):
    script_path = SRC_DIR / filename
    print(f"\n=== Running {script_path.name} ===")
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    run_script("data_loader.py")
    run_script("models_historical.py")
    run_script("models_monte_carlo.py")
    run_script("models_vix_regression.py")
    run_script("evaluation_kupiec.py")
    run_script("evaluation_summary.py")
    print("\nAll steps completed.")
