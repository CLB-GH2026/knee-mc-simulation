"""
Single-subject (OKS004) interactive viewer — 650 nm.

See "Knee Model_Tissue On_Off_808nm.py" for why this is now a thin wrapper
around the batch script's run_subject() instead of an independent copy of
the pipeline (the 650nm viewer previously had its own cone-angle constant
that disagreed with the batch script's, among other drifted config).
"""

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BATCH_PATH = _HERE / "OKS Knee Models_MC Results_650nm.py"

_spec = importlib.util.spec_from_file_location("_oks_knee_650nm_batch", _BATCH_PATH)
_batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_batch)

SUBJECT_ID = "OKS004"

if __name__ == "__main__":
    output_dir = _HERE / "viewer_results_650nm"
    output_dir.mkdir(exist_ok=True)

    for condition in _batch.MELANIN_CONDITIONS:
        print(f"\n{'=' * 60}")
        print(f"  Viewer run: {SUBJECT_ID} @ 650nm — {condition.upper()}")
        print(f"{'=' * 60}")
        _batch.run_subject(SUBJECT_ID, _HERE, output_dir, melanin_condition=condition)
