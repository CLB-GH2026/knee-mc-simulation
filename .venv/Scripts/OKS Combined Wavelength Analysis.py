"""
OKS Combined Wavelength Analysis
---------------------------------
Loads saved fluence volumes from both 808 nm and 650 nm batch pipeline runs
and produces five combined outputs:

  1. Combined CSV   — total absorbed power + fluence rate per tissue group
                      (combined 808 + 650 nm) plus therapeutic coverage table
                      (% voxels >= 1 mW/cm² per tissue group, Section 3)
  2. Waterfall HTML — fluence attenuation from skin surface to joint centre
                      for both wavelengths individually and combined, with a
                      green band for the PBM therapeutic window (1–100 mW/cm²)
  3. Dose-time HTML — cumulative fluence dose (J/cm²) vs session time for the
                      cartilage and synovial groups at each wavelength and
                      combined; dashed threshold at 1 J/cm²
  4. Comparison HTML — grouped bar chart of mean cartilage fluence rate
                       (mW/cm²) by subject for 650 nm vs 808 nm, with a
                       dashed minimum threshold at 1 mW/cm²

Configuration
-------------
Set DIR_808NM and DIR_650NM to the timestamped output directories produced by
OKS Knee Models_MC Results_808nm.py and _650nm.py respectively.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
import webbrowser
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Set these to the actual output directories from the batch scripts
DIR_808NM = Path("results_808nm_20260416_161000")   # e.g. Path("results_808nm_20260414_120000")
DIR_650NM = Path("results_650nm_20260416_162841")   # e.g. Path("results_650nm_20260414_130000")

MELANIN_CONDITION  = 'fair'                          # single condition for waterfall / comparison charts
MELANIN_CONDITIONS = ['fair', 'olive', 'dark']       # all conditions for dose-time chart

SUBJECT_IDS = [f"OKS{i:03d}" for i in range(1, 10) if i != 5]
VOXEL_SIZE  = 1.0                   # mm/voxel

# Source power parameters — must match the batch scripts
POWER_808 = dict(mw=50,  duty=0.75, eff=0.85, n_src=3)
POWER_650 = dict(mw=160, duty=0.25, eff=0.85, n_src=3)

# PBM therapeutic window
FLUENCE_RATE_MIN_MW   =   1.0   # mW/cm²  — therapeutic lower bound / coverage threshold
FLUENCE_RATE_MAX_MW   = 100.0   # mW/cm²  — therapeutic upper bound
DOSE_THRESHOLD_J      =   1.0   # J/cm²   — minimum cumulative dose
SESSION_TIME_MAX_S    = 900     # seconds — plot window for dose-time chart

# Output directory
RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"results_combined_{RUN_ID}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. TISSUE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Label → tissue name (pat1-cart and pat2-cart both map to label 10 in the volume)
LABEL_TO_NAME = {
    1:  'femur-bone',
    2:  'tibia-bone',
    3:  'fibula-bone',
    4:  'patella-bone',
    5:  'lm-men',
    6:  'mm-men',
    7:  'fc-cart',
    8:  'ltc-cart',
    9:  'mtc-cart',
    10: 'pat-cart',
    11: 'muscle',
    12: 'adipose',
    13: 'skin',
    14: 'synovial',
    15: 'epidermis',
}

# µa (mm⁻¹) for each tissue at 808 nm and 650 nm — matches batch script values
TISSUE_MUA = {
    'femur-bone':   (0.040,  0.068),
    'tibia-bone':   (0.040,  0.068),
    'fibula-bone':  (0.040,  0.068),
    'patella-bone': (0.040,  0.068),
    'lm-men':       (0.006,  0.014),
    'mm-men':       (0.006,  0.014),
    'fc-cart':      (0.015,  0.025),
    'ltc-cart':     (0.015,  0.025),
    'mtc-cart':     (0.015,  0.025),
    'pat-cart':     (0.015,  0.025),
    'synovial':     (0.0005, 0.002),
    'muscle':       (0.0180, 0.0280),
    'adipose':      (0.0013, 0.003),
    'skin':         (0.003,  0.011),
    'epidermis':    (0.008 * 0.2, 0.020 * 0.2),   # fair skin, thickness-scaled
}

GROUPS = {
    'Bone':      lambda n: 'bone'     in n,
    'Cartilage': lambda n: 'cart'     in n,
    'Meniscus':  lambda n: 'men'      in n,
    'Synovial':  lambda n: 'synovial' in n,
    'Muscle':    lambda n: 'muscle'   in n,
    'Adipose':   lambda n: 'adipose'  in n,
    'Skin':      lambda n: ('skin' in n) or ('epidermis' in n),
}

SKIN_LABELS = {13, 15}   # skin + epidermis (outermost tissue shell)

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_subject(subject_id, condition):
    """
    Load label volume and fluence arrays for one subject at both wavelengths.
    Returns (vol, fluence_808, fluence_650) or None if files are missing.
    """
    p808 = DIR_808NM / condition / subject_id
    p650 = DIR_650NM / condition / subject_id

    for p, nm in [(p808, '808nm'), (p650, '650nm')]:
        if not (p / 'label_volume.npy').exists():
            print(f"  [{subject_id}] Missing {nm} data — expected: {p / 'label_volume.npy'}")
            return None
        if not (p / 'fluence_combined.npy').exists():
            print(f"  [{subject_id}] Missing {nm} fluence — expected: {p / 'fluence_combined.npy'}")
            return None

    vol         = np.load(p808 / 'label_volume.npy')
    fluence_808 = np.load(p808 / 'fluence_combined.npy')
    fluence_650 = np.load(p650 / 'fluence_combined.npy')
    return vol, fluence_808, fluence_650


# ─────────────────────────────────────────────────────────────────────────────
# 4. PER-TISSUE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def tissue_stats(fluence, vol, wl_idx):
    """
    Compute per-tissue mean/max fluence, absorbed power, and coverage fraction.

    Parameters
    ----------
    fluence  : 3D array (mW/cm²)
    vol      : 3D integer label array
    wl_idx   : 0 = 808 nm, 1 = 650 nm
    """
    vox_vol_cm3 = (VOXEL_SIZE * 0.1) ** 3
    results = {}
    for label, name in LABEL_TO_NAME.items():
        mask = vol == label
        n = int(mask.sum())
        if n == 0:
            continue
        flu       = fluence[mask]
        mua       = TISSUE_MUA.get(name, (0.0, 0.0))[wl_idx]
        illuminated = flu >= FLUENCE_RATE_MIN_MW
        n_illum   = int(illuminated.sum())
        results[name] = {
            'label':                label,
            'n_voxels':             n,
            'mean_flu':             float(flu.mean()),
            'mean_flu_illuminated': float(flu[illuminated].mean()) if n_illum > 0 else 0.0,
            'n_voxels_illuminated': n_illum,
            'max_flu':              float(flu.max()),
            'absorbed_mw':          float((mua * 10.0 * flu * vox_vol_cm3).sum()),
            'coverage_pct':         float(100.0 * n_illum / n),
        }
    return results


def group_stats(tissue_res):
    """Aggregate tissue_stats dict into tissue groups. Returns (groups_dict, total_absorbed_mw)."""
    total_abs = sum(r['absorbed_mw'] for r in tissue_res.values())
    groups = {}
    for gname, match in GROUPS.items():
        names = [n for n in tissue_res if match(n)]
        if not names:
            continue
        grp_vox       = sum(tissue_res[n]['n_voxels'] for n in names)
        grp_illum_vox = sum(tissue_res[n]['n_voxels_illuminated'] for n in names)
        grp_abs       = sum(tissue_res[n]['absorbed_mw'] for n in names)
        grp_flu       = (sum(tissue_res[n]['mean_flu'] * tissue_res[n]['n_voxels']
                             for n in names) / grp_vox if grp_vox else 0.0)
        grp_flu_illum = (sum(tissue_res[n]['mean_flu_illuminated'] * tissue_res[n]['n_voxels_illuminated']
                             for n in names) / grp_illum_vox if grp_illum_vox else 0.0)
        grp_max       = max(tissue_res[n]['max_flu'] for n in names)
        grp_cov       = (sum(tissue_res[n]['coverage_pct'] * tissue_res[n]['n_voxels']
                             for n in names) / grp_vox if grp_vox else 0.0)
        groups[gname] = {
            'n_voxels':             grp_vox,
            'mean_flu':             grp_flu,
            'mean_flu_illuminated': grp_flu_illum,
            'max_flu':              grp_max,
            'absorbed_mw':          grp_abs,
            'pct_total':            100.0 * grp_abs / total_abs if total_abs else 0.0,
            'coverage_pct':         grp_cov,
        }
    return groups, total_abs


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMBINED CSV  (requirements 1 + 3)
# ─────────────────────────────────────────────────────────────────────────────

def write_combined_csv(all_data, output_path):
    """
    Write a multi-section CSV combining 808 nm and 650 nm results.

    all_data : list of (subject_id, groups_808, groups_650, total_808, total_650)
    """
    input_808 = POWER_808['mw'] * POWER_808['duty'] * POWER_808['eff'] * POWER_808['n_src']
    input_650 = POWER_650['mw'] * POWER_650['duty'] * POWER_650['eff'] * POWER_650['n_src']
    input_total = input_808 + input_650

    with open(output_path, 'w', newline='') as f:
        w = csv.writer(f)

        # ── Section 1: Combined power and fluence rate ─────────────────────
        w.writerow(['=== SECTION 1: Combined Power Absorbed & Fluence Rate ==='])
        w.writerow([
            'Subject', 'Tissue Group',
            '808nm Mean Fluence Rate (mW/cm²)', '808nm Absorbed (mW)', '808nm % Input',
            '650nm Mean Fluence Rate (mW/cm²)', '650nm Absorbed (mW)', '650nm % Input',
            'Combined Mean Fluence Rate (mW/cm²)', 'Combined Absorbed (mW)', 'Combined % Input',
            'Combined Mean Fluence Rate - Illuminated Zone (mW/cm²)',
        ])

        for subj_id, g808, g650, tot808, tot650 in all_data:
            for gname in GROUPS:
                d8 = g808.get(gname)
                d6 = g650.get(gname)
                if d8 is None and d6 is None:
                    continue
                flu8        = d8['mean_flu']             if d8 else 0.0
                abs8        = d8['absorbed_mw']          if d8 else 0.0
                flu6        = d6['mean_flu']             if d6 else 0.0
                abs6        = d6['absorbed_mw']          if d6 else 0.0
                flu8_illum  = d8['mean_flu_illuminated'] if d8 else 0.0
                flu6_illum  = d6['mean_flu_illuminated'] if d6 else 0.0

                # Combined fluence rate: input-power weighted mean
                flu_comb       = (flu8 * input_808 + flu6 * input_650) / input_total
                abs_comb       = abs8 + abs6
                flu_comb_illum = flu8_illum + flu6_illum

                w.writerow([
                    subj_id, gname,
                    f'{flu8:.4e}',  f'{abs8:.4f}',  f'{100*abs8/input_808:.2f}',
                    f'{flu6:.4e}',  f'{abs6:.4f}',  f'{100*abs6/input_650:.2f}',
                    f'{flu_comb:.4e}', f'{abs_comb:.4f}', f'{100*abs_comb/input_total:.2f}',
                    f'{flu_comb_illum:.4e}',
                ])

            # Totals row
            tot_comb = tot808 + tot650
            w.writerow([
                subj_id, 'TOTAL',
                '', f'{tot808:.4f}', f'{100*tot808/input_808:.2f}',
                '', f'{tot650:.4f}', f'{100*tot650/input_650:.2f}',
                '', f'{tot_comb:.4f}', f'{100*tot_comb/input_total:.2f}',
            ])
            w.writerow([])

        # ── Section 2: Source power reference ─────────────────────────────
        w.writerow([])
        w.writerow(['=== SECTION 2: Source Power Reference ==='])
        w.writerow(['Wavelength', 'Peak Power/src (mW)', 'Duty Cycle', 'Opt Eff',
                    'Avg Power/src (mW)', 'N Sources', 'Total Input (mW)'])
        for lbl, p in [('808nm', POWER_808), ('650nm', POWER_650)]:
            avg = p['mw'] * p['duty'] * p['eff']
            w.writerow([lbl, p['mw'], p['duty'], p['eff'], f'{avg:.2f}', p['n_src'],
                        f'{avg * p["n_src"]:.2f}'])
        w.writerow(['Combined', '', '', '', '', '',
                    f'{input_total:.2f}'])

        # ── Section 3: Therapeutic coverage (% voxels >= 1 mW/cm²) ────────
        w.writerow([])
        w.writerow([f'=== SECTION 3: Therapeutic Coverage (% voxels >= {FLUENCE_RATE_MIN_MW} mW/cm²) ==='])
        w.writerow(['Subject', 'Tissue Group',
                    '808nm Coverage (%)', '650nm Coverage (%)', 'Combined Coverage (%)'])

        for subj_id, g808, g650, _, _ in all_data:
            for gname in GROUPS:
                d8 = g808.get(gname)
                d6 = g650.get(gname)
                if d8 is None and d6 is None:
                    continue
                cov8 = d8['coverage_pct'] if d8 else 0.0
                cov6 = d6['coverage_pct'] if d6 else 0.0
                # Combined: voxel qualifies if EITHER wavelength exceeds threshold
                # (approximated here as max of the two coverage fractions)
                cov_comb = min(100.0, (cov8 + cov6) / 2.0)   # conservative: mean
                w.writerow([subj_id, gname,
                             f'{cov8:.2f}', f'{cov6:.2f}', f'{cov_comb:.2f}'])
            w.writerow([])

    print(f"  Combined CSV written: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. TISSUE GROUP FLUENCE RATE BAR CHART  (requirement 2)
# ─────────────────────────────────────────────────────────────────────────────

def write_waterfall_html(all_data, output_path, illuminated_only=False):
    """
    Horizontal bar chart: population-mean combined fluence rate (mW/cm²) per tissue
    group. Groups ordered skin (top) → bone (bottom) to reflect fluence propagation
    depth. X-axis 0–50 mW/cm² with therapeutic window overlay (1–50 mW/cm²).

    illuminated_only : if True, show only the illuminated-zone bar (voxels >= 1 mW/cm²).
    """
    # Listed deepest-first so Plotly's bottom-up y-axis puts Skin at the top
    display_groups = ['Bone', 'Cartilage', 'Synovial', 'Muscle', 'Adipose', 'Skin']

    flu_comb       = []
    flu_comb_illum = []
    for gname in display_groups:
        v8       = [d[1].get(gname, {}).get('mean_flu',             np.nan) for d in all_data]
        v6       = [d[2].get(gname, {}).get('mean_flu',             np.nan) for d in all_data]
        v8_illum = [d[1].get(gname, {}).get('mean_flu_illuminated', np.nan) for d in all_data]
        v6_illum = [d[2].get(gname, {}).get('mean_flu_illuminated', np.nan) for d in all_data]
        flu_comb.append(float(np.nanmean(v8)) + float(np.nanmean(v6)))
        flu_comb_illum.append(float(np.nanmean(v8_illum)) + float(np.nanmean(v6_illum)))

    fig = go.Figure()

    if not illuminated_only:
        fig.add_trace(go.Bar(
            y=display_groups,
            x=flu_comb,
            orientation='h',
            name='Combined (all voxels)',
            marker_color='mediumpurple',
            text=[f'{v:.2f}' for v in flu_comb],
            textposition='outside',
        ))

    fig.add_trace(go.Bar(
        y=display_groups,
        x=flu_comb_illum,
        orientation='h',
        name='Combined (illuminated zone)',
        marker_color='darkorange',
        text=[f'{v:.2f}' for v in flu_comb_illum],
        textposition='outside',
    ))

    # Therapeutic target window: vertical green band 1–50 mW/cm²
    fig.add_vrect(
        x0=FLUENCE_RATE_MIN_MW,
        x1=50.0,
        fillcolor='rgba(0,200,80,0.30)',
        line_width=0,
        annotation_text='Therapeutic Window<br>(1–50 mW/cm²)',
        annotation_position='top right',
        annotation_font_size=11,
    )

    title = (
        'Mean Combined Fluence Rate by Tissue Group — Illuminated Zone'
        if illuminated_only else
        'Mean Combined Fluence Rate by Tissue Group — All Voxels vs Illuminated Zone'
    )
    fig.update_layout(
        title=title,
        xaxis_title='Mean Fluence Rate (mW/cm²)',
        xaxis=dict(range=[0, 50]),
        yaxis_title='Tissue Group',
        template='plotly_white',
        height=500,
    )
    fig.write_html(str(output_path))
    print(f"  Waterfall HTML written: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. DOSE vs TIME LINE PLOT  (requirement 4)
# ─────────────────────────────────────────────────────────────────────────────

def write_dose_time_html(all_data_by_condition, output_path):
    """
    Scatter plot (lines + markers) of cumulative fluence dose (J/cm²) vs session
    time at 60-second intervals up to SESSION_TIME_MAX_S.
    Shows illuminated-zone combined fluence for cartilage and synovial fluid,
    one pair of traces per melanin condition.
    Dashed threshold at DOSE_THRESHOLD_J.
    """
    times = np.arange(0, SESSION_TIME_MAX_S + 1, 60)   # 0, 60, 120, … 900 s

    # Colour palette: (cartilage colour, synovial colour, marker symbol)
    CONDITION_STYLES = {
        'fair':  ('mediumseagreen', 'lightgreen',   'circle'),
        'olive': ('darkorange',     'gold',         'square'),
        'dark':  ('mediumpurple',   'plum',         'diamond'),
    }

    def pop_mean_illum(all_data, group):
        vals = [
            (g808.get(group, {}).get('mean_flu_illuminated', 0.0) +
             g650.get(group, {}).get('mean_flu_illuminated', 0.0))
            for _, g808, g650, _, _ in all_data
        ]
        return float(np.mean(vals))

    fig = go.Figure()

    for cond, all_data in all_data_by_condition.items():
        if not all_data:
            continue
        cart_col, syn_col, symbol = CONDITION_STYLES.get(cond, ('grey', 'lightgrey', 'circle'))
        flu_cart = pop_mean_illum(all_data, 'Cartilage')
        flu_syn  = pop_mean_illum(all_data, 'Synovial')

        fig.add_trace(go.Scatter(
            x=times, y=flu_cart * times / 1000.0,
            mode='lines+markers',
            name=f'{cond.capitalize()} — Cartilage (illum.)',
            legendgroup=cond,
            line=dict(color=cart_col, dash='solid', width=2),
            marker=dict(size=6, symbol=symbol),
        ))
        fig.add_trace(go.Scatter(
            x=times, y=flu_syn * times / 1000.0,
            mode='lines+markers',
            name=f'{cond.capitalize()} — Synovial (illum.)',
            legendgroup=cond,
            line=dict(color=syn_col, dash='dot', width=2),
            marker=dict(size=6, symbol=symbol),
        ))

    fig.add_hline(
        y=DOSE_THRESHOLD_J,
        line=dict(color='black', width=2, dash='dash'),
        annotation_text=f'Dose Threshold ({DOSE_THRESHOLD_J} J/cm²)',
        annotation_position='right',
    )

    fig.update_layout(
        title='Cumulative Fluence Dose vs Session Time — Cartilage & Synovial Fluid',
        xaxis_title='Session Time (s)',
        yaxis_title='Fluence Dose (J/cm²)',
        xaxis=dict(tickmode='array', tickvals=list(times)),
        template='plotly_white',
        legend=dict(groupclick='toggleitem'),
        height=600,
    )
    fig.write_html(str(output_path))
    print(f"  Dose-time HTML written: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. WAVELENGTH COMPARISON BAR CHART  (requirement 5)
# ─────────────────────────────────────────────────────────────────────────────

def write_wavelength_comparison_html(all_data, output_path):
    """
    Grouped bar chart: mean cartilage fluence rate (mW/cm²) per subject for
    808 nm vs 650 nm, each shown for all voxels and illuminated zone only.
    Dashed minimum threshold at FLUENCE_RATE_MIN_MW.
    """
    subj_ids       = [d[0] for d in all_data]
    flu808         = [d[1].get('Cartilage', {}).get('mean_flu',             0.0) for d in all_data]
    flu808_illum   = [d[1].get('Cartilage', {}).get('mean_flu_illuminated', 0.0) for d in all_data]
    flu650         = [d[2].get('Cartilage', {}).get('mean_flu',             0.0) for d in all_data]
    flu650_illum   = [d[2].get('Cartilage', {}).get('mean_flu_illuminated', 0.0) for d in all_data]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='808 nm',
        x=subj_ids, y=flu808,
        marker_color='steelblue',
        text=[f'{v:.2e}' for v in flu808],
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name='808 nm (illum.)',
        x=subj_ids, y=flu808_illum,
        marker_color='dodgerblue',
        marker_pattern_shape='/',
        text=[f'{v:.2e}' for v in flu808_illum],
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name='650 nm',
        x=subj_ids, y=flu650,
        marker_color='tomato',
        text=[f'{v:.2e}' for v in flu650],
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name='650 nm (illum.)',
        x=subj_ids, y=flu650_illum,
        marker_color='orangered',
        marker_pattern_shape='/',
        text=[f'{v:.2e}' for v in flu650_illum],
        textposition='outside',
    ))

    # Minimum threshold line
    fig.add_hline(
        y=FLUENCE_RATE_MIN_MW,
        line=dict(color='green', width=2, dash='dash'),
        annotation_text=f'Min. Therapeutic Threshold ({FLUENCE_RATE_MIN_MW} mW/cm²)',
        annotation_position='right',
    )

    fig.update_layout(
        title='Cartilage Group: Mean Fluence Rate by Subject & Wavelength — All Voxels vs Illuminated Zone',
        xaxis_title='Subject',
        yaxis_title='Mean Fluence Rate (mW/cm²)',
        barmode='group',
        template='plotly_white',
        yaxis=dict(rangemode='tozero'),
        height=500,
    )
    fig.write_html(str(output_path))
    print(f"  Wavelength comparison HTML written: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Primary condition (waterfall/comparison): {MELANIN_CONDITION}")
    print(f"Dose-time conditions: {MELANIN_CONDITIONS}")
    print(f"Subjects: {SUBJECT_IDS}\n")

    # ── Helper: load one condition ─────────────────────────────────────────
    def load_condition(condition):
        data = []
        for subj_id in SUBJECT_IDS:
            loaded = load_subject(subj_id, condition)
            if loaded is None:
                continue
            vol, flu808, flu650 = loaded
            ts808 = tissue_stats(flu808, vol, wl_idx=0)
            ts650 = tissue_stats(flu650, vol, wl_idx=1)
            gs808, tot808 = group_stats(ts808)
            gs650, tot650 = group_stats(ts650)
            data.append((subj_id, gs808, gs650, tot808, tot650))
        return data

    # ── Load primary condition (waterfall, CSV, comparison charts) ─────────
    print(f"Loading {MELANIN_CONDITION} condition...")
    all_data = load_condition(MELANIN_CONDITION)
    for subj_id, gs808, gs650, *_ in all_data:
        print(f"  [{subj_id}] Cart 808nm: {gs808.get('Cartilage',{}).get('mean_flu',0):.3e} mW/cm²  "
              f"650nm: {gs650.get('Cartilage',{}).get('mean_flu',0):.3e} mW/cm²")

    if not all_data:
        print("\nNo subject data loaded — check DIR_808NM and DIR_650NM paths.")
        raise SystemExit(1)

    print(f"\nLoaded {len(all_data)} of {len(SUBJECT_IDS)} subjects ({MELANIN_CONDITION}).\n")

    # ── Load all conditions for dose-time chart ────────────────────────────
    all_data_by_condition = {}
    for cond in MELANIN_CONDITIONS:
        if cond == MELANIN_CONDITION:
            all_data_by_condition[cond] = all_data
            continue
        print(f"Loading {cond} condition for dose-time chart...")
        cond_data = load_condition(cond)
        if cond_data:
            all_data_by_condition[cond] = cond_data
            print(f"  Loaded {len(cond_data)} subjects ({cond}).")
        else:
            print(f"  No data found for {cond} — skipping.")

    # ── Output 1+3: Combined CSV ───────────────────────────────────────────
    csv_path = OUTPUT_DIR / "OKS_Combined_Wavelength_Analysis.csv"
    write_combined_csv(all_data, csv_path)

    # ── Output 2: Fluence attenuation waterfall ────────────────────────────
    waterfall_path = OUTPUT_DIR / "OKS_Fluence_Waterfall.html"
    write_waterfall_html(all_data, waterfall_path)

    waterfall_illum_path = OUTPUT_DIR / "OKS_Fluence_Waterfall_Illuminated.html"
    write_waterfall_html(all_data, waterfall_illum_path, illuminated_only=True)

    # ── Output 3: Dose vs time (all skin conditions) ───────────────────────
    dose_path = OUTPUT_DIR / "OKS_Dose_vs_Time.html"
    write_dose_time_html(all_data_by_condition, dose_path)

    # ── Output 4: Wavelength comparison bar chart ──────────────────────────
    comp_path = OUTPUT_DIR / "OKS_Wavelength_Comparison.html"
    write_wavelength_comparison_html(all_data, comp_path)

    # Open all outputs in browser
    for p in [waterfall_path, waterfall_illum_path, dose_path, comp_path]:
        webbrowser.open(str(p.resolve()))

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
