from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd


RNG = np.random.default_rng(2398412)


def get_entry(row, heavy=True):
    entry = {}
    suffix = "heavy" if heavy else "light"
    seq = ""
    start = 0
    for region in ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3"]:
        # Build up sequence through concatenation
        cur = row[f"{region}_aa_{suffix}"]
        seq += cur
        
        # Keep track of boundaries
        length = len(cur)
        entry[f"{region}_start"] = start
        entry[f"{region}_end"] = start + length
        start += length

    # Hacks to get framework 4
    full_seq = row[f"sequence_alignment_aa_{suffix}"]
    fwr4_aa = full_seq[start:]
    seq += fwr4_aa
        
    entry[f"fwr4_start"] = start
    entry[f"fwr4_end"] = start + len(fwr4_aa)
        
    # OCD check to ensure dumped indices work as intended
    for region in ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3"]:
        cur = row[f"{region}_aa_{suffix}"]
        start = entry[f"{region}_start"]
        end = entry[f"{region}_end"]
        assert seq[start:end] == cur, f"Failed on {region}"

    entry["seq"] = seq
    return entry


def extract_data(filepath):
    # Skip metadata included by OAS
    x = pd.read_csv(filepath, skiprows=[0])
    
    # Shuffle by resampling whole thing
    x = x.sample(frac=1, random_state=RNG.bit_generator)
    
    # Require both heavy and light chains to be good
    x = x[(x.productive_heavy == "T") & (x.vj_in_frame_heavy == "T") & (x.productive_light == "T") & (x.vj_in_frame_light == "T")]
    df = x.iloc[:100]
    
    entries = defaultdict(dict)
    # Rely on some OAS filename conventions to name
    root = filepath.stem.split("_")[0]
    for ab, row in df.iterrows():
        name = f"{root}_{ab}"
        entries[name]["heavy"] = get_entry(row, heavy=True)
        entries[name]["light"] = get_entry(row, heavy=False)

    return entries


if __name__ == "__main__":
    val_dir = Path("val_data")

    all_data = {}
    for filepath in val_dir.glob("*.csv"):
        x = extract_data(filepath)
        all_data = {**all_data, **x}

    with open("val_data.json", "w") as f:
        json.dump(all_data, f)