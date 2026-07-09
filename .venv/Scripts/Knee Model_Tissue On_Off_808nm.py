"""
Single-subject (OKS004) interactive viewer — 808 nm.

This used to be an independent ~1000-line copy of the full pipeline (its own
build_label_volume/run_pmcx/plot_results/etc.), which is exactly the kind of
duplication that let it drift from the batch script: it wrapped different
soft-tissue thicknesses, different source positions, and a power reference
that didn't even match its own run_pmcx call (a hardcoded "50 * 0.75" print
alongside a `source_power_mw=25` default).

Since the batch script's run_subject() already does the full single-subject
pipeline (build → simulate → analyze → plot → write HTML → open browser), the
correct fix is to call that directly rather than maintain a second copy. This
script does nothing but that, for OKS004 across all three melanin conditions.
"""

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BATCH_PATH = _HERE / "OKS Knee Models_MC Results_808nm.py"

_spec = importlib.util.spec_from_file_location("_oks_knee_808nm_batch", _BATCH_PATH)
_batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_batch)  # module __name__ != "__main__", so its own
                                   # main block does not run; only defs/config load.

SUBJECT_ID = "OKS004"

if __name__ == "__main__":
    output_dir = _HERE / "viewer_results_808nm"
    output_dir.mkdir(exist_ok=True)

    for condition in _batch.MELANIN_CONDITIONS:
        print(f"\n{'=' * 60}")
        print(f"  Viewer run: {SUBJECT_ID} @ 808nm — {condition.upper()}")
        print(f"{'=' * 60}")
        _batch.run_subject(SUBJECT_ID, _HERE, output_dir, melanin_condition=condition)
