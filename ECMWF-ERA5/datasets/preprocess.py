from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_era5_land_chengyu.py"


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")