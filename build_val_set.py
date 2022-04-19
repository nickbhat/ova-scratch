from collections import defaultdict
import json
import itertools
from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd

from anarci import anarci


RNG = np.random.default_rng(2398412)


def get_entry(row, heavy=True):
    entry = {}
    prefix = "HC" if heavy else "LC"
    oas_suffix = "heavy" if heavy else "light"
    seq = ""
    start = 0
    for region in ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3"]:
        # Build up sequence through concatenation
        cur = row[f"{region}_aa_{oas_suffix}"]
        seq += cur
        
        # Keep track of boundaries
        length = len(cur)
        entry[f"{prefix}_{region}_start"] = start
        entry[f"{prefix}_{region}_end"] = start + length - 1
        start += length

    # Hacks to get framework 4
    full_seq = row[f"sequence_alignment_aa_{oas_suffix}"]
    fwr4_aa = full_seq[start:]
    seq += fwr4_aa
        
    entry[f"{prefix}_fwr4_start"] = start
    entry[f"{prefix}_fwr4_end"] = start + len(fwr4_aa) - 1
        
    # OCD check to ensure dumped indices work as intended
    for region in ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3"]:
        cur = row[f"{region}_aa_{oas_suffix}"]
        start = entry[f"{prefix}_{region}_start"]
        end = entry[f"{prefix}_{region}_end"]
        assert seq[start:end + 1] == cur, f"Failed on {region}"

    entry[prefix] = seq
    
    # Get alignment info
    if heavy:
        raw = row.ANARCI_numbering_heavy
    else:
        raw = row.ANARCI_numbering_light
    # Flip single and double quotes
    num, _, _ = anarci([("f ", seq)], scheme="aho")
    num = num[0][0][0]  # lol
    aho_nums = []
    for (n, _), aa in num:
        if aa != "-":
            aho_nums.append(n)
    assert(len(aho_nums) == len(seq)), "Lengths did not match."
    entry[f"{prefix}_AHo"] = aho_nums
    return entry


def sample_df(filepath):
    # Skip metadata included by OAS
    x = pd.read_csv(filepath, skiprows=[0])
    
    # Shuffle by resampling whole thing
    x = x.sample(frac=1, random_state=RNG.bit_generator)
    
    # Require both heavy and light chains to be good
    x = x[(x.productive_heavy == "T") & (x.vj_in_frame_heavy == "T") & (x.productive_light == "T") & (x.vj_in_frame_light == "T")]
    df = x.iloc[:100]
    
    entries = []
    # Rely on some OAS filename conventions to name
    root = filepath.stem.split("_")[0]
    for ab, row in df.iterrows():
        name = f"{root}_{ab}"
        hc_entry = get_entry(row, heavy=True)
        lc_entry = get_entry(row, heavy=False)
        entry = {**hc_entry, **lc_entry}
        entry["Antibody"] = name
        entries.append(entry)

    return entries


def extract_data(filepath, val_indices):
    # This function loads from existing val names because I was having random seed issues..    
    # Skip metadata included by OAS
    x = pd.read_csv(filepath, skiprows=[0])
    x = x[(x.productive_heavy == "T") & (x.vj_in_frame_heavy == "T") & (x.productive_light == "T") & (x.vj_in_frame_light == "T")]
    
    # Shuffle by resampling whole thing
    df = x.iloc[val_indices]
    
    entries = []
    # Rely on some OAS filename conventions to name
    for ab, row in df.iterrows():
        name = f"{root}_{ab}"
        heavy_entry = get_entry(row, heavy=True)
        light_entry = get_entry(row, heavy=False)
        entry = {**heavy_entry, **light_entry}
        entry["Antibody"] = name
        entries.append(entry)

    return entries



if __name__ == "__main__":
    val_dir = Path("val_data")
    with open(val_dir / Path("val_names.pkl"), "rb") as f:
        val_names = pkl.load(f)
    idxs = defaultdict(list)
    for name in val_names:
        root = name.split("_")[0]
        idx = int(name.split("_")[1])
        idxs[root].append(idx)
        
    all_data = []
    for filepath in val_dir.glob("*.csv"):
        root = filepath.stem.split("_")[0]
        x = extract_data(filepath, idxs[root])
        all_data.extend(x)

    with open("val_data.json", "w") as f:
        json.dump(all_data, f)