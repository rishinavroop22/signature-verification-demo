"""All folder locations in one place. Set SIGVER_ROOT to move the heavy files."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SIGVER_ROOT", Path(__file__).resolve().parents[1] / "models"))
DATA = ROOT / "data"
CEDAR_SRC = DATA / "cedar_src"            # the team's original 224x224 images and CSVs
CEDAR = DATA / "cedar_clean"              # cleaned 128x256 images
SID = DATA / "sid_clean"
SPLITS = DATA / "splits"
GENERATED = DATA / "generated"            # generated/<model>/<split>/...
RUNS = ROOT / "runs"                      # checkpoints, logs, sample grids
RESULTS = Path(__file__).resolve().parents[1] / "results"

RATIOS = ["70_30", "60_40", "80_20"]
WRITER_COUNTS = {"70_30": (30, 8, 17), "60_40": (26, 7, 22), "80_20": (35, 9, 11)}
GENERATORS = ["cgan", "wgan_gp", "ddpm"]
IMG_H, IMG_W = 128, 256
SEED = 42
