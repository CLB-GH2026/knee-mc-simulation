"""
Single-subject (OKS004) interactive viewer — legacy unsuffixed entry point.

Predates the 650nm/808nm wavelength split; kept as a thin wrapper pointed at
the 808nm batch script for backward compatibility. Prefer
"Knee Model_Tissue On_Off_808nm.py" or "..._650nm.py" directly for new work.

See "Knee Model_Tissue On_Off_808nm.py" for why this is a thin wrapper around
the batch script's run_subject() instead of an independent copy of the
pipeline.
"""

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BATCH_PATH = _HERE / "OKS Knee Models_MC Results_808nm.py"

_spec = importlib.util.spec_from_file_location("_oks_knee_808nm_batch", _BATCH_PATH)
_batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_batch)

SUBJECT_ID = "OKS004"

if __name__ == "__main__":
    output_dir = _HERE / "viewer_results_808nm"
    output_dir.mkdir(exist_ok=True)

    for condition in _batch.MELANIN_CONDITIONS:
        print(f"\n{'=' * 60}")
        print(f"  Viewer run: {SUBJECT_ID} @ 808nm — {condition.upper()}")
        print(f"{'=' * 60}")
        _batch.run_subject(SUBJECT_ID, _HERE, output_dir, melanin_condition=condition)
