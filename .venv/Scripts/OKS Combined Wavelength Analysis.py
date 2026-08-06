"""
OKS Combined Wavelength Analysis
--------------------------------
Cross-subject combined 650+808 analysis for the knee. Reads the per-wavelength
fluence volumes saved by the combined device run (OKS Knee Models_MC Results_
Combined.py): fluence_808.npy / fluence_650.npy / label_volume.npy in one folder
per subject/condition, auto-detecting the latest results_knee_combined_* dir.
Produces:

  1. Combined CSV      — per subject/group: 808 / 650 / combined mean fluence +
                         absorbed power + % input, plus combined illumination-
                         zone coverage and fluence, per melanin condition.
  2. Waterfall HTML    — population-mean fluence by tissue group (all voxels vs
                         illuminated zone), fair skin.
  3. Dose-vs-time HTML — population-mean illuminated-zone dose (J/cm²) vs session
                         time for cartilage / meniscus / synovial, per condition.
  4. Comparison HTML   — mean cartilage fluence by subject (808 / 650 / combined).

Consistency: the illumination zone is defined on the COMBINED field
(f_808 + f_650 >= 1 mW/cm²) — the device delivers both wavelengths at once —
matching pbm_mc_core.analyze_combined_absorption used in the runs. (Previously
this script read separate per-wavelength batch dirs and thresholded each
wavelength independently, which inflated the combined illumination numbers.)
"""

import numpy as np
import plotly.graph_objects as go
import csv
from pathlib import Path
from datetime import datetime

BASE = Path(".")
SUBJECT_IDS = [f"OKS{i:03d}" for i in range(1, 10) if i != 5]
MELANIN_CONDITIONS = ["fair", "olive", "dark"]

FLUENCE_RATE_MIN_MW = 1.0
SESSION_TIME_MAX_S = 900
DOSE_THRESHOLD_J = 1.0

# Knee source power (matches the combined run: 808 = 50 mW/0.75, 650 = 160 mW/0.25).
POWER_808 = dict(mw=50,  duty=0.75, eff=0.85, n_src=3)
POWER_650 = dict(mw=160, duty=0.25, eff=0.85, n_src=3)

LABEL_TO_NAME = {
    1: 'femur-bone', 2: 'tibia-bone', 3: 'fibula-bone', 4: 'patella-bone',
    5: 'lm-men', 6: 'mm-men', 7: 'fc-cart', 8: 'ltc-cart', 9: 'mtc-cart',
    10: 'pat-cart', 11: 'muscle', 12: 'adipose', 13: 'skin', 14: 'synovial',
    15: 'epidermis',
}
TISSUE_MUA = {  # µa (mm⁻¹) at (808, 650); epidermis handled per-condition below.
    'femur-bone': (0.040, 0.068), 'tibia-bone': (0.040, 0.068),
    'fibula-bone': (0.040, 0.068), 'patella-bone': (0.040, 0.068),
    'lm-men': (0.006, 0.014), 'mm-men': (0.006, 0.014),
    'fc-cart': (0.015, 0.025), 'ltc-cart': (0.015, 0.025),
    'mtc-cart': (0.015, 0.025), 'pat-cart': (0.015, 0.025),
    'muscle': (0.0180, 0.0280), 'adipose': (0.0013, 0.003),
    'skin': (0.003, 0.011), 'synovial': (0.0005, 0.002),
}
_EPI_SCALE = 0.2
MELANIN_EPI = {
    'fair':  (0.008 * _EPI_SCALE, 0.020 * _EPI_SCALE),
    'olive': (0.025 * _EPI_SCALE, 0.070 * _EPI_SCALE),
    'dark':  (0.075 * _EPI_SCALE, 0.200 * _EPI_SCALE),
}
GROUPS = {
    'Bone':           lambda n: 'bone'     in n,
    'Cartilage':      lambda n: 'cart'     in n,
    'Meniscus':       lambda n: 'men'      in n,
    'Synovial':       lambda n: 'synovial' in n,
    'Muscle':         lambda n: 'muscle'   in n,
    'Adipose':        lambda n: 'adipose'  in n,
    'Skin+Epidermis': lambda n: ('skin' in n) or ('epidermis' in n),
}
TARGET_GROUPS = ['Cartilage', 'Meniscus', 'Synovial']
VOXEL_SIZE = 1.0


def _latest(pattern):
    dirs = sorted(BASE.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def load(combined_dir, condition, subject):
    p = combined_dir / condition / subject
    if not (p / "fluence_808.npy").exists():
        return None
    return (np.load(p / "label_volume.npy"),
            np.load(p / "fluence_808.npy"),
            np.load(p / "fluence_650.npy"))


def combined_stats(f8, f6, vol, condition):
    """Per-tissue stats; illumination zone on the COMBINED field (f8+f6)."""
    vox_cm3 = (VOXEL_SIZE * 0.1) ** 3
    fc = f8 + f6
    out = {}
    for label, name in LABEL_TO_NAME.items():
        mask = vol == label
        n = int(mask.sum())
        if n == 0:
            continue
        a8, a6, ac = f8[mask], f6[mask], fc[mask]
        mua8, mua6 = MELANIN_EPI[condition] if name == 'epidermis' else TISSUE_MUA.get(name, (0.0, 0.0))
        illum = ac >= FLUENCE_RATE_MIN_MW
        n_ill = int(illum.sum())
        out[name] = {
            'n_voxels': n,
            'mean_808': float(a8.mean()), 'mean_650': float(a6.mean()), 'mean_comb': float(ac.mean()),
            'absorbed_808': float((mua8 * 10.0 * a8 * vox_cm3).sum()),
            'absorbed_650': float((mua6 * 10.0 * a6 * vox_cm3).sum()),
            'n_illuminated': n_ill,
            'coverage_pct': 100.0 * n_ill / n,
            'illum_fluence': float(ac[illum].mean()) if n_ill else 0.0,
        }
        out[name]['absorbed_comb'] = out[name]['absorbed_808'] + out[name]['absorbed_650']
    return out


def group_stats(ts):
    g = {}
    for name, match in GROUPS.items():
        ks = [k for k in ts if match(k)]
        if not ks:
            continue
        vox = sum(ts[k]['n_voxels'] for k in ks)
        ivox = sum(ts[k]['n_illuminated'] for k in ks)
        wm = lambda f: (sum(ts[k][f] * ts[k]['n_voxels'] for k in ks) / vox) if vox else 0.0
        g[name] = {
            'n_voxels': vox,
            'mean_808': wm('mean_808'), 'mean_650': wm('mean_650'), 'mean_comb': wm('mean_comb'),
            'absorbed_808': sum(ts[k]['absorbed_808'] for k in ks),
            'absorbed_650': sum(ts[k]['absorbed_650'] for k in ks),
            'absorbed_comb': sum(ts[k]['absorbed_comb'] for k in ks),
            'coverage_pct': wm('coverage_pct'),
            'illum_fluence': (sum(ts[k]['illum_fluence'] * ts[k]['n_illuminated'] for k in ks) / ivox) if ivox else 0.0,
        }
    return g


# ── Outputs ──────────────────────────────────────────────────────────────────
def write_combined_csv(all_data, out_path):
    in8 = POWER_808['mw'] * POWER_808['duty'] * POWER_808['eff'] * POWER_808['n_src']
    in6 = POWER_650['mw'] * POWER_650['duty'] * POWER_650['eff'] * POWER_650['n_src']
    itot = in8 + in6
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'=== Combined 650+808 by Subject & Tissue Group '
                    f'(illumination zone = combined field >= {FLUENCE_RATE_MIN_MW} mW/cm2) ==='])
        w.writerow(['Condition', 'Subject', 'Group',
                    '808 Mean Fluence (mW/cm2)', '650 Mean Fluence (mW/cm2)', 'Combined Mean Fluence (mW/cm2)',
                    'Combined Absorbed (mW)', 'Combined % Input',
                    'Illum. Coverage (%)', 'Illum. Fluence (mW/cm2)'])
        for cond in MELANIN_CONDITIONS:
            for subj, g in all_data.get(cond, {}).items():
                for gname in GROUPS:
                    d = g.get(gname)
                    if d is None:
                        continue
                    w.writerow([cond, subj, gname,
                                f'{d["mean_808"]:.4e}', f'{d["mean_650"]:.4e}', f'{d["mean_comb"]:.4e}',
                                f'{d["absorbed_comb"]:.4f}', f'{100 * d["absorbed_comb"] / itot:.2f}',
                                f'{d["coverage_pct"]:.2f}', f'{d["illum_fluence"]:.4e}'])
                w.writerow([])
        w.writerow(['Source power', f'808: {in8:.1f} mW', f'650: {in6:.1f} mW', f'Combined: {itot:.1f} mW'])
    print(f"  CSV: {out_path}")


def _pop_group(all_data, condition, field):
    """Population-mean of a per-group field across subjects for a condition."""
    subs = all_data.get(condition, {})
    out = {}
    for gname in GROUPS:
        vals = [g[gname][field] for g in subs.values() if gname in g]
        out[gname] = float(np.mean(vals)) if vals else 0.0
    return out


def write_waterfall(all_data, out_path):
    display = ['Bone', 'Cartilage', 'Meniscus', 'Synovial', 'Muscle', 'Adipose', 'Skin+Epidermis']
    allm = _pop_group(all_data, 'fair', 'mean_comb')
    illm = _pop_group(all_data, 'fair', 'illum_fluence')
    fig = go.Figure()
    fig.add_trace(go.Bar(y=display, x=[allm.get(x, 0) for x in display], orientation='h',
                         name='All voxels (combined mean)', marker_color='mediumpurple',
                         text=[f'{allm.get(x,0):.2f}' for x in display], textposition='outside'))
    fig.add_trace(go.Bar(y=display, x=[illm.get(x, 0) for x in display], orientation='h',
                         name='Illuminated zone (>=1 mW/cm²)', marker_color='darkorange',
                         text=[f'{illm.get(x,0):.2f}' for x in display], textposition='outside'))
    fig.add_vrect(x0=FLUENCE_RATE_MIN_MW, x1=50.0, fillcolor='rgba(0,200,80,0.25)', line_width=0,
                  annotation_text='Therapeutic window (1–50 mW/cm²)', annotation_position='top right')
    fig.update_layout(title='Knee — Population-mean Combined Fluence by Tissue Group (fair, n=%d)' % len(all_data.get('fair', {})),
                      xaxis_title='Mean Fluence Rate (mW/cm²)', yaxis_title='Tissue Group',
                      template='plotly_white', height=520, barmode='group')
    fig.write_html(str(out_path))
    print(f"  Waterfall: {out_path}")


def write_dose_time(all_data, out_path):
    times = np.arange(0, SESSION_TIME_MAX_S + 1, 60)
    styles = {'fair': ('seagreen', 'solid'), 'olive': ('darkorange', 'dash'), 'dark': ('mediumpurple', 'dot')}
    fig = go.Figure()
    for cond in MELANIN_CONDITIONS:
        if cond not in all_data:
            continue
        illm = _pop_group(all_data, cond, 'illum_fluence')
        col, dash = styles.get(cond, ('grey', 'solid'))
        for grp, sym in zip(TARGET_GROUPS, ['circle', 'square', 'diamond']):
            flu = illm.get(grp, 0.0)
            fig.add_trace(go.Scatter(x=times, y=flu * times / 1000.0, mode='lines+markers',
                                     name=f'{cond} — {grp}', legendgroup=cond,
                                     line=dict(color=col, dash=dash, width=2), marker=dict(size=4, symbol=sym)))
    fig.add_hline(y=DOSE_THRESHOLD_J, line=dict(color='black', width=2, dash='dash'),
                  annotation_text=f'{DOSE_THRESHOLD_J} J/cm² threshold', annotation_position='right')
    fig.update_layout(title='Knee — Illuminated-zone Dose vs Session Time (population mean)',
                      xaxis_title='Session Time (s)', yaxis_title='Cumulative Dose (J/cm²)',
                      template='plotly_white', height=560, legend=dict(groupclick='toggleitem'))
    fig.write_html(str(out_path))
    print(f"  Dose-vs-time: {out_path}")


def write_comparison(all_data, out_path):
    subs = list(all_data.get('fair', {}).keys())
    def cart(subj, key):
        return all_data['fair'][subj].get('Cartilage', {}).get(key, 0.0)
    fig = go.Figure()
    for wl, key, col in [('808 nm', 'mean_808', 'steelblue'),
                         ('650 nm', 'mean_650', 'tomato'),
                         ('Combined', 'mean_comb', 'seagreen')]:
        fig.add_trace(go.Bar(name=wl, x=subs, y=[cart(s, key) for s in subs], marker_color=col))
    fig.add_hline(y=FLUENCE_RATE_MIN_MW, line=dict(color='green', width=2, dash='dash'),
                  annotation_text='1 mW/cm² threshold', annotation_position='right')
    fig.update_layout(title='Knee — Cartilage Mean Fluence by Subject & Wavelength (fair)',
                      xaxis_title='Subject', yaxis_title='Mean Fluence Rate (mW/cm²)',
                      barmode='group', template='plotly_white', height=500)
    fig.write_html(str(out_path))
    print(f"  Comparison: {out_path}")


if __name__ == "__main__":
    combined_dir = _latest("results_knee_combined_*")
    if combined_dir is None:
        raise SystemExit("No results_knee_combined_* directory found — run the knee combined script first.")
    print(f"Reading combined run: {combined_dir}")

    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT = Path(f"results_knee_combined_analysis_{RUN_ID}")
    OUT.mkdir(parents=True, exist_ok=True)

    all_data = {}   # condition -> {subject -> group_stats}
    for cond in MELANIN_CONDITIONS:
        for subj in SUBJECT_IDS:
            loaded = load(combined_dir, cond, subj)
            if loaded is None:
                continue
            vol, f8, f6 = loaded
            all_data.setdefault(cond, {})[subj] = group_stats(combined_stats(f8, f6, vol, cond))
        n = len(all_data.get(cond, {}))
        if n:
            cart = np.mean([g.get('Cartilage', {}).get('coverage_pct', 0) for g in all_data[cond].values()])
            print(f"  [{cond}] {n} subjects | mean cartilage illum. coverage {cart:.1f}%")

    if not all_data:
        raise SystemExit("No subject data loaded.")

    write_combined_csv(all_data, OUT / "OKS_Combined_Wavelength.csv")
    write_waterfall(all_data, OUT / "OKS_Fluence_Waterfall.html")
    write_dose_time(all_data, OUT / "OKS_Dose_vs_Time.html")
    write_comparison(all_data, OUT / "OKS_Wavelength_Comparison.html")
    print(f"\nDone. All outputs in {OUT}")
