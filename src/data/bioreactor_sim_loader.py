'''Unpickle bioreactor simulation datasets from CNODE/datasets/bioreactor_sim_by_noise
and write each one to a CSV file in the same directory.'''

import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # CNODE/
DATA_PATH = PROJECT_ROOT / "datasets" / "bioreactor_sim_by_noise"


def unpickle_to_csv():
    pkl_files = sorted(DATA_PATH.glob("*.pkl"))
    assert pkl_files, f"No .pkl files found in {DATA_PATH.absolute()}"

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)
        csv_path = pkl_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path.name}  ({len(df)} rows)")


if __name__ == "__main__":
    unpickle_to_csv()
