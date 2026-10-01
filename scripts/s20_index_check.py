"""The scenario 20 index-file defect, measured from the released files: results/s20_index.json.

The index file names each label file by its `index` column; the label files are numbered by row. The column jumps at
the start of sequence 4, so from there on the named files either do not exist or belong to other frames.

    python scripts/s20_index_check.py
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("DEEPSENSE_ROOT", r"N:\Datasets\DeepSense 6G\extracted")) / "scenario20"


def main():
    import pandas as pd
    df = pd.read_csv(RAW / "scenario20.csv")
    jump = df["index"].diff()
    at = int(jump.idxmax())
    s4 = df[df["seq_index"] == 4]
    exists = [(RAW / x.lstrip("./")).exists() for x in s4["unit1_blockage"]]
    out = {"rows": len(df), "jump_at_row": at + 1, "jump": int(jump.max()), "seq4_rows": len(s4),
           "seq4_named_missing": int(len(exists) - sum(exists)), "seq4_named_existing": int(sum(exists)),
           "label_files": len([f for f in os.listdir(RAW / "unit1" / "label_data") if f.endswith(".txt")])}
    lab = {r["target"]: r for r in json.loads((ROOT / "results" / "label_agreement.json").read_text())}
    out["seq4_by_row_agreement"] = lab["scenario20"]["per_seq"]["4"]["frame"]["agreement"]
    (ROOT / "results" / "s20_index.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
