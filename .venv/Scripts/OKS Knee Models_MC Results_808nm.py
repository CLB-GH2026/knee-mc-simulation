"""
STL Mesh → 3D Voxel Volume + pmcx Fluence Overlay (808 nm)
--------------------------------------------------
Pipeline (shared logic lives in pbm_mc_core; see that package's README for the
full stage list and the tissue-label convention this script's `tissues` dict
follows):
  1. Load one STL per tissue, voxelize via ray casting → integer label volume
  2. Build & run pmcx simulation on that volume
  3. Load fluence output, log-transform
  4. Render with Plotly:
       - One Isosurface per tissue (semi-transparent, colored by tissue)
       - Fluence Volume (Isosurface colormap, log scale) overlaid on top

Dependencies:
    pip install numpy trimesh pmcx plotly scipy
    pip install git+https://github.com/CLB-GH2026/pbm-mc-core.git@v0.1.0
"""

import numpy as np
import time
import webbrowser
from pathlib import Path
from datetime import datetime

from pbm_mc_core import (
    opt, EPIDERMIS_LABEL, build_melanin_conditions,
    build_label_volume,
    add_synovial_fluid, add_wrapping_layers, add_epidermis_layer,
    find_joint_line_z, find_surface_source_positions,
    optimize_source_positions_reciprocity,
    run_pmcx,
    analyze_fluence_absorption, analyze_penetration_depth, plot_depth_histogram,
    target_depth_zone,
    plot_results, write_interactive_html,
    results_to_csv, melanin_comparison_to_csv,
    ensure_repo_current,
)

base_dir = Path()
mesh_dir = base_dir / 'Raw_Mesh_Files_OKS004'

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

start_time = time.perf_counter()

WAVELENGTH_M = 808e-9
WAVELENGTH_NM = 808

# Epidermal optical properties by melanin condition at 808 nm.
# Melanin absorption follows λ^-3.33 — values ~4× lower than at 650 nm.
# True (unscaled) values; build_melanin_conditions() applies the epidermis
# thickness-correction scale (0.2 mm physical / 1 mm voxel).
_MELANIN_RAW_808NM = {
    #        µa      µs'    g     n
    'fair':  (0.008, 1.50, 0.80, 1.40),  # Fitzpatrick I-II,   f_mel ~1.3%
    'olive': (0.025, 1.60, 0.80, 1.40),  # Fitzpatrick III-IV, f_mel ~4.4%
    'dark':  (0.075, 1.70, 0.80, 1.40),  # Fitzpatrick V-VI,   f_mel ~16%
}

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE OPTIMIZATION CONFIG
# Set OPTIMIZE_SOURCES = True to run a reciprocity scan before each subject.
# ─────────────────────────────────────────────────────────────────────────────
OPTIMIZE_SOURCES = False   # True → per-subject reciprocity scan before main run
OPT_N_SOURCES    = 3       # number of source positions to find
OPT_MIN_SEP_MM   = 25.0    # minimum separation between selected positions (mm)
OPT_NPHOTON      = 1e6     # photons for optimization run (less than main run)

VOXEL_SIZE = 1.0               # mm per voxel
GRID_DIMS_MM = (150, 140, 285)   # x, y, z in mm — edit these, not VOXEL_RES
VOXEL_RES = tuple(int(round(d / VOXEL_SIZE)) for d in GRID_DIMS_MM)

FLUENCE_OUTPUT  = True   # None to run pmcx, or True to load saved files
AUTO_ORIENT     = True   # auto-detect and correct Z-axis inversion (OKS002-type)
AUTO_OPEN_HTML  = True   # open each fluence overlay in the browser as it is written

# Soft-tissue wrapping layer thicknesses (mm).
MUSCLE_THICK_MM  = 8
ADIPOSE_THICK_MM = 4
SKIN_THICK_MM    = 2

# Source power parameters — single definition used by run_pmcx, analyze_fluence_absorption,
# and results_to_csv so that all three are guaranteed consistent.
SOURCE_POWER_MW   = 50     # peak power per source (mW)
SOURCE_DUTY_CYCLE = 0.75   # modulation duty cycle
SOURCE_OPT_EFF    = 0.85   # optical coupling efficiency
CONE_ANGLE_DEG    = 20     # source cone full angle

MELANIN_CONDITIONS = build_melanin_conditions(_MELANIN_RAW_808NM, voxel_size_mm=VOXEL_SIZE)

# ─────────────────────────────────────────────────────────────────────────────
# TISSUE GROUPS (knee anatomy) — passed into analyze_fluence_absorption /
# results_to_csv / melanin_comparison_to_csv, which are anatomy-agnostic.
# ─────────────────────────────────────────────────────────────────────────────
GROUPS = {
    'Bone':      lambda n: 'bone'     in n,
    'Cartilage': lambda n: 'cart'     in n,
    'Meniscus':  lambda n: 'men'      in n,
    'Synovial':  lambda n: 'synovial' in n,
    'Muscle':    lambda n: 'muscle'   in n,
    'Adipose':   lambda n: 'adipose'  in n,
    'Skin+Epidermis': lambda n: ('skin' in n) or ('epidermis' in n),
}
DOSE_GROUPS = {
    'Cartilage':      lambda n: 'cart'     in n,
    'Muscle':         lambda n: 'muscle'   in n,
    'Synovial Fluid': lambda n: 'synovial' in n,
}
COMP_GROUPS = {
    'Cartilage':      lambda n: 'cart'     in n,
    'Synovial Fluid': lambda n: 'synovial' in n,
    'Muscle':         lambda n: 'muscle'   in n,
    'Bone':           lambda n: 'bone'     in n,
    'Skin+Epidermis': lambda n: 'skin'     in n or 'epidermis' in n,
}

# ─────────────────────────────────────────────────────────────────────────────
# TISSUE COLORS
# ─────────────────────────────────────────────────────────────────────────────
TISSUE_COLORS = {
    1:  "rgba(128,128,128,1.00)",   # femur-bone
    2:  "rgba(128,128,128,1.00)",   # tibia-bone
    3:  "rgba(128,128,128,1.00)",   # fibula-bone
    4:  "rgba(128,128,128,1.00)",   # patella-bone
    5:  "rgba(0,0,255,1.00)",       # lm-men
    6:  "rgba(0,0,255,1.00)",       # mm-men
    7:  "rgba(0,0,255,1.00)",       # fc-cart
    8:  "rgba(0,0,255,1.00)",       # ltc-cart
    9:  "rgba(0,0,255,1.00)",       # mtc-cart
    10: "rgba(0,0,255,1.00)",       # pat-cart
    11: "rgba(180,60,60,0.4)",      # muscle
    12: "rgba(255,220,150,0.4)",    # adipose
    13: "rgba(210,180,140,0.30)",   # skin        — overridden per melanin condition
    14: "rgba(173,216,230,0.5)",    # synovial
    15: "rgba(255,228,196,0.15)",   # epidermis   — overridden per melanin condition
}

SKIN_TONE_COLORS = {
    #           label-13 (skin/dermis)          label-15 (epidermis)
    'fair':  {13: "rgba(255,213,170,0.30)",  15: "rgba(255,220,185,0.15)"},  # Fitzpatrick I-II
    'olive': {13: "rgba(185,130,85,0.30)",   15: "rgba(198,143,95,0.15)"},   # Fitzpatrick III-IV
    'dark':  {13: "rgba(101,60,28,0.30)",    15: "rgba(115,72,35,0.15)"},    # Fitzpatrick V-VI
}


def run_subject(subject_id, mesh_dir_base, output_dir, melanin_condition='fair'):
    """
    Run the full pipeline for a single subject.
    Returns (subject_id, results) or None if failed.
    """
    mesh_dir = Path(mesh_dir_base) / f"Raw_Mesh_Files_{subject_id}"

    if not mesh_dir.exists():
        print(f"  Skipping {subject_id} — directory not found: {mesh_dir}")
        return None

    print(f"\n{'=' * 60}")
    print(f"  Processing {subject_id}")
    print(f"{'=' * 60}")

    tissues = {
        "synovial":     (None,                                            14, opt(0.0005, 0.01,  0.90, 1.36)),  # water-like fluid
        "skin":         (None,                                            13, opt(0.003,  1.22,  0.79, 1.40)),
        "adipose":      (None,                                            12, opt(0.0013, 1.00,  0.90, 1.44)),
        "muscle":       (None,                                            11, opt(0.0180, 0.55,  0.93, 1.37)),
        "pat1-cart":    (mesh_dir / "patella_lig_raw.stl",                10, opt(0.015,  1.50,  0.90, 1.37)),  # fibrocartilage/ligament
        "pat2-cart":    (mesh_dir / "patella_cartilage_raw.stl",          10, opt(0.015,  1.00,  0.90, 1.37)),  # hyaline
        "mtc-cart":     (mesh_dir / "tibia_cartilage_med_raw.stl",         9, opt(0.015,  1.00,  0.90, 1.37)),
        "ltc-cart":     (mesh_dir / "tibia_cartilage_lat_raw.stl",         8, opt(0.015,  1.00,  0.90, 1.37)),
        "fc-cart":      (mesh_dir / "femur_cartilage_raw.stl",             7, opt(0.015,  1.00,  0.90, 1.37)),
        "mm-men":       (mesh_dir / "men_med_raw.stl",                     6, opt(0.006,  1.80,  0.90, 1.37)),
        "lm-men":       (mesh_dir / "men_lat_raw.stl",                     5, opt(0.006,  1.80,  0.90, 1.37)),
        "patella-bone": (mesh_dir / "patella_raw.stl",                     4, opt(0.040,  2.50,  0.92, 1.37)),
        "fibula-bone":  (mesh_dir / "fibula_raw.stl",                      3, opt(0.040,  2.50,  0.92, 1.37)),
        "tibia-bone":   (mesh_dir / "tibia_raw.stl",                       2, opt(0.040,  2.50,  0.92, 1.37)),
        "femur-bone":   (mesh_dir / "femur_raw.stl",                       1, opt(0.040,  2.50,  0.92, 1.37)),
    }
    tissues["epidermis"] = (None, EPIDERMIS_LABEL, MELANIN_CONDITIONS[melanin_condition])

    try:
        # ── Step 1: Build label volume ────────────────────────────────────
        vol, origin, mesh_center = build_label_volume(
            tissues, VOXEL_RES, VOXEL_SIZE,
            auto_orient=AUTO_ORIENT,
            orient_ref_a='femur-bone', orient_ref_b='tibia-bone',
        )

        # ── Step 2: Add synovial fluid and wrapping layers ───────────────
        bone_labels      = [t[1] for name, t in tissues.items() if "bone" in name]
        cartilage_labels = [t[1] for name, t in tissues.items() if "cart" in name]
        meniscus_labels  = [t[1] for name, t in tissues.items() if "men"  in name]

        vol = add_synovial_fluid(
            vol,
            cartilage_labels=cartilage_labels + meniscus_labels,
            bone_labels=bone_labels,
            fluid_label=tissues["synovial"][1],
            dilation_vox=3
        )

        layer_configs_vox = [
            (tissues["muscle"][1],  int(round(MUSCLE_THICK_MM  / VOXEL_SIZE))),
            (tissues["adipose"][1], int(round(ADIPOSE_THICK_MM / VOXEL_SIZE))),
            (tissues["skin"][1],    int(round(SKIN_THICK_MM    / VOXEL_SIZE))),
        ]
        vol = add_wrapping_layers(vol, layer_configs_vox)
        vol = add_epidermis_layer(vol, skin_label=tissues["skin"][1],
                                   epidermis_label=EPIDERMIS_LABEL)

        # ── Step 2b: Locate joint line Z ─────────────────────────────────
        jl_z = find_joint_line_z(vol, tissues, origin, VOXEL_SIZE, mesh_center)

        # ── Step 3: Compute source directions and place on epidermis surface
        _colors = ['red', 'green', 'blue', 'orange', 'purple']
        if OPTIMIZE_SOURCES:
            print("\n--- Reciprocity source position optimisation ---")
            opt_positions = optimize_source_positions_reciprocity(
                vol, tissues, origin, mesh_center, VOXEL_SIZE,
                OPT_N_SOURCES, OPT_MIN_SEP_MM, OPT_NPHOTON,
                epidermis_label=EPIDERMIS_LABEL,
            )
            if opt_positions:
                src_configs = [
                    {'name': f'Opt-{i+1}', 'world_pos': pos,
                     'color': _colors[i % len(_colors)]}
                    for i, pos in enumerate(opt_positions)
                ]
            else:
                print("  [OPT] Falling back to default positions")
                src_configs = [
                    {'name': 'Posterior',    'world_pos': [  0, -60, jl_z], 'color': 'red'  },
                    {'name': 'Anterior (L)', 'world_pos': [-30,  55, jl_z], 'color': 'green'},
                    {'name': 'Anterior (R)', 'world_pos': [ 30,  55, jl_z], 'color': 'blue' },
                ]
        else:
            src_configs = [
                {'name': 'Posterior',    'world_pos': [  0, -60, jl_z], 'color': 'red'  },
                {'name': 'Anterior (L)', 'world_pos': [-30,  55, jl_z], 'color': 'green'},
                {'name': 'Anterior (R)', 'world_pos': [ 30,  55, jl_z], 'color': 'blue' },
            ]
        for cfg in src_configs:
            d = np.array([0, 0, jl_z]) - np.array(cfg['world_pos'])
            cfg['srcdir'] = (d / np.linalg.norm(d)).tolist()

        pmcx_source_plus = find_surface_source_positions(
            vol, origin, VOXEL_SIZE, mesh_center, src_configs
        )
        pmcx_source = [{'srcpos': s['srcpos'], 'srcdir': s['srcdir']}
                       for s in pmcx_source_plus]

        # ── Step 4: Run pmcx ──────────────────────────────────────────────
        fluence_combined, fluence_list = run_pmcx(
            vol, tissues, pmcx_source,
            wavelength_m=WAVELENGTH_M,
            source_power_mw=SOURCE_POWER_MW,
            duty_cycle=SOURCE_DUTY_CYCLE,
            opt_eff=SOURCE_OPT_EFF,
            cone_angle_deg=CONE_ANGLE_DEG,
            voxel_size_mm=VOXEL_SIZE,
        )

        # ── Step 6: Absorption analysis ───────────────────────────────────
        results = analyze_fluence_absorption(
            fluence_combined, vol, tissues, VOXEL_SIZE,
            pmcx_source=pmcx_source,
            groups=GROUPS,
            source_power_mw=SOURCE_POWER_MW,
            duty_cycle=SOURCE_DUTY_CYCLE,
            opt_eff=SOURCE_OPT_EFF,
        )

        # ── Step 7: Save subject outputs ──────────────────────────────────
        subj_dir = Path(output_dir) / melanin_condition / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        cart_names  = [n for n in results if 'cart' in n]
        cart_vox    = sum(results[n]['n_voxels'] for n in cart_names)
        cart_flu_mw = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                           for n in cart_names) / cart_vox) if cart_vox > 0 else 0.0

        syn_names   = [n for n in results if 'synovial' in n]
        syn_vox     = sum(results[n]['n_voxels'] for n in syn_names)
        syn_flu_mw  = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                           for n in syn_names) / syn_vox) if syn_vox > 0 else 0.0

        try:
            print("\n=== Penetration depth analysis ===")
            bin_centers, mean_flu, max_depth = analyze_penetration_depth(
                fluence_combined, vol, VOXEL_SIZE, mesh_center, origin
            )
            z_lo, z_hi, z_med = target_depth_zone(
                vol, tissues, VOXEL_SIZE,
                lambda n: ('cart' in n) or ('synovial' in n) or ('men' in n))
            if z_lo is None:
                z_lo, z_hi, z_med = 2.0, 3.5, 2.75
            print(f"  Target depth zone: {z_lo:.2f}-{z_hi:.2f} cm (median {z_med:.2f} cm)")
            fig_depth = plot_depth_histogram(
                bin_centers, mean_flu, subject_id, WAVELENGTH_NM,
                depth_refs=[(z_med, 'Cartilage/meniscus/synovial (targets)')],
                zone_lo=z_lo, zone_hi=z_hi,
                cartilage_flu_mw=cart_flu_mw,
                synovial_flu_mw=syn_flu_mw,
            )
            depth_html = str(subj_dir / f"depth_histogram_{subject_id}_{melanin_condition}.html")
            fig_depth.write_html(depth_html)
            print(f"  Saved: {depth_html}")
        except Exception as _depth_err:
            print(f"  WARNING: penetration depth analysis skipped: {_depth_err}")

        np.save(subj_dir / "label_volume.npy", vol)
        np.save(subj_dir / "fluence_combined.npy", fluence_combined)
        for i, flu in enumerate(fluence_list):
            np.save(subj_dir / f"fluence_src{i + 1}.npy", flu)

        # ── Step 7b: 3D fluence overlay HTML ──────────────────────────────
        all_fluences   = [fluence_combined] + fluence_list
        fluence_names_local = ['Combined'] + [f'Source {i+1}' for i in range(len(fluence_list))]

        tissue_colors = {**TISSUE_COLORS, **SKIN_TONE_COLORS.get(melanin_condition, {})}
        fig = plot_results(
            vol, fluence_combined, fluence_list,
            all_fluences, fluence_names_local,
            tissues, origin, VOXEL_SIZE,
            pmcx_source_plus=pmcx_source_plus,
            tissue_colors=tissue_colors,
            mesh_center=mesh_center,
        )
        overlay_html = str(subj_dir / f"fluence_overlay_{subject_id}_{melanin_condition}.html")
        write_interactive_html(
            fig, tissues, all_fluences, fluence_names_local,
            pmcx_source_plus,
            output_path=overlay_html,
        )
        print(f"  Saved: {overlay_html}")
        if AUTO_OPEN_HTML:
            webbrowser.open(f"file:///{Path(overlay_html).resolve().as_posix()}")

        return subject_id, results

    except Exception as e:
        print(f"  ERROR processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    ensure_repo_current(Path(__file__).resolve().parent.parent.parent)

    start_time  = time.perf_counter()
    SUBJECT_IDS = [f"OKS{i:03d}" for i in range(1, 10) if i != 5]
    BASE_DIR    = Path(__file__).resolve().parent
    RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR  = Path(f"results_808nm_{RUN_ID}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    print(f"Processing {len(SUBJECT_IDS)} subjects × "
          f"{len(MELANIN_CONDITIONS)} melanin conditions")
    print(f"Subjects: {SUBJECT_IDS}")

    all_condition_results = {}

    for condition in MELANIN_CONDITIONS:
        print(f"\n{'=' * 60}")
        print(f"  Melanin condition: {condition.upper()}")
        print(f"{'=' * 60}")
        (OUTPUT_DIR / condition).mkdir(exist_ok=True)

        cond_results = []
        for subject_id in SUBJECT_IDS:
            result = run_subject(subject_id, BASE_DIR, OUTPUT_DIR,
                                 melanin_condition=condition)
            if result is not None:
                cond_results.append(result)

        all_condition_results[condition] = cond_results

        if cond_results:
            csv_path = OUTPUT_DIR / f"MC_Analysis_808nm_{condition}.csv"
            results_to_csv(
                cond_results,
                groups=GROUPS,
                dose_groups=DOSE_GROUPS,
                source_power_mw=SOURCE_POWER_MW,
                duty_cycle=SOURCE_DUTY_CYCLE,
                opt_eff=SOURCE_OPT_EFF,
                n_sources=3,
                output_path=str(csv_path),
            )
            print(f"  Completed {len(cond_results)} of {len(SUBJECT_IDS)} subjects")

    melanin_comparison_to_csv(
        all_condition_results,
        groups=COMP_GROUPS,
        output_path=str(OUTPUT_DIR / "MC_Melanin_Comparison_808nm.csv"),
        wavelength_nm=WAVELENGTH_NM,
    )

    end_time = time.perf_counter()
    print(f"\nTotal elapsed: {end_time - start_time:.2f} seconds")
