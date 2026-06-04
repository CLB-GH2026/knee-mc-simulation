"""
STL Mesh → 3D Voxel Volume + pmcx Fluence Overlay
--------------------------------------------------
Pipeline:
  1. Load one STL per tissue, voxelize via ray casting → integer label volume
  2. Build & run pmcx simulation on that volume
  3. Load fluence output, log-transform
  4. Render with Plotly:
       - One Isosurface per tissue (semi-transparent, colored by tissue)
       - Fluence Volume (Isosurface colormap, log scale) overlaid on top

Dependencies:
    pip install numpy trimesh pmcx plotly scipy
"""

import numpy as np
import trimesh
import time
import pmcx
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, distance_transform_edt
from pathlib import Path
import webbrowser
import os
from datetime import datetime

base_dir = Path()
mesh_dir = base_dir / 'Raw_Mesh_Files_OKS004'

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

start_time = time.perf_counter()

def opt(mua, mus_prime, g, n):
    """Convert reduced scattering coefficient to transport scattering for pmcx."""
    return [mua, mus_prime / (1 - g), g, n]


# Epidermal optical properties by melanin condition at 808 nm
# Melanin absorption follows λ^-3.33 — values ~4× lower than at 650 nm.
#
# Thickness correction: epidermis is 0.2 mm physically but occupies 1 voxel
# (1 mm) in the simulation.  Both µa and µs' are scaled by
#   EPI_SCALE = EPI_THICKNESS_MM / VOXEL_SIZE_MM = 0.2 / 1.0 = 0.2
# so that a photon crossing the 1 mm voxel experiences the same total
# absorption and scattering as it would traversing the real 0.2 mm layer.
# g and n are dimensionless and are NOT scaled.
#
# Expected single-pass absorption (Beer-Lambert, normal incidence):
#   fair:  1 - exp(-0.008 × 0.2) ≈ 0.16%
#   olive: 1 - exp(-0.025 × 0.2) ≈ 0.50%
#   dark:  1 - exp(-0.075 × 0.2) ≈ 1.49%
_EPI_THICKNESS_MM = 0.2
_EPI_SCALE        = _EPI_THICKNESS_MM / 1.0   # divide by VOXEL_SIZE (1 mm)

MELANIN_CONDITIONS = {
    #                   µa (true × scale)        µs' (true × scale)   g     n
    'fair':  opt(0.008 * _EPI_SCALE, 1.50 * _EPI_SCALE, 0.80, 1.40),  # Fitzpatrick I-II,   f_mel ~1.3%
    'olive': opt(0.025 * _EPI_SCALE, 1.60 * _EPI_SCALE, 0.80, 1.40),  # Fitzpatrick III-IV, f_mel ~4.4%
    'dark':  opt(0.075 * _EPI_SCALE, 1.70 * _EPI_SCALE, 0.80, 1.40),  # Fitzpatrick V-VI,   f_mel ~16%
}
EPIDERMIS_LABEL = 15

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE OPTIMIZATION CONFIG
# Set OPTIMIZE_SOURCES = True to run a reciprocity scan before each subject.
# The scan places an isotropic virtual source at the cartilage+synovial centroid
# and finds which epidermis voxels receive the most photons — those are the
# reciprocally optimal external illumination sites (optical reciprocity theorem).
# OPT_NPHOTON can be much smaller than the main run since only the surface
# hotspot locations matter, not absolute fluence accuracy.
# ─────────────────────────────────────────────────────────────────────────────
OPTIMIZE_SOURCES = False   # True → per-subject reciprocity scan before main run
OPT_N_SOURCES    = 3       # number of source positions to find
OPT_MIN_SEP_MM   = 25.0    # minimum separation between selected positions (mm)
OPT_NPHOTON      = 1e6     # photons for optimization run (less than main run)


def run_subject(subject_id, mesh_dir_base, output_dir, melanin_condition='fair'):
    """
    Run the full pipeline for a single subject.
    Returns absorption results dict or None if failed.
    """
    mesh_dir = Path(mesh_dir_base) / f"Raw_Mesh_Files_{subject_id}"

    if not mesh_dir.exists():
        print(f"  Skipping {subject_id} — directory not found: {mesh_dir}")
        return None

    print(f"\n{'=' * 60}")
    print(f"  Processing {subject_id}")
    print(f"{'=' * 60}")

    # Update TISSUES paths for this subject (Updated properties: 04142026)
    tissues = {
        "synovial":     (None,                                            14, opt(0.0005, 0.01,  0.90, 1.36)),  # corrected: water-like fluid
        "skin":         (None,                                            13, opt(0.003,  1.22,  0.79, 1.40)),
        "adipose":      (None,                                            12, opt(0.0013, 1.00,  0.90, 1.44)),  # µs' corrected from 0.20
        "muscle":       (None,                                            11, opt(0.0180, 0.55,  0.93, 1.37)),
        "pat1-cart":    (mesh_dir / "patella_lig_raw.stl",                10, opt(0.015,  1.50,  0.90, 1.37)),  # upper range fibrocartilage/ligament
        "pat2-cart":    (mesh_dir / "patella_cartilage_raw.stl",          10, opt(0.015,  1.00,  0.90, 1.37)),  # upper range hyaline
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
    # Epidermis: outermost 0.2 mm layer — optical props vary by melanin condition
    tissues["epidermis"] = (None, EPIDERMIS_LABEL, MELANIN_CONDITIONS[melanin_condition])

    try:
        # ── Step 1: Build label volume ────────────────────────────────────
        vol, origin, mesh_center = build_label_volume(tissues, VOXEL_RES, VOXEL_SIZE)

        # ── Step 2: Add synovial fluid and wrapping layers ───────────────
        # NOTE: wrapping must happen BEFORE source placement so that sources
        # are snapped to the outer epidermis surface, not to bare bone/cartilage.
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
        # All source world_pos Z values are set to jl_z so that each source
        # sits at the height of maximum cartilage/synovial density, minimising
        # oblique path length from source to joint regardless of subject anatomy.
        jl_z = find_joint_line_z(vol, tissues, origin, VOXEL_SIZE, mesh_center)

        # ── Step 3: Compute source directions and place on epidermis surface
        _colors = ['red', 'green', 'blue', 'orange', 'purple']
        if OPTIMIZE_SOURCES:
            print("\n--- Reciprocity source position optimisation ---")
            opt_positions = optimize_source_positions_reciprocity(
                vol, tissues, origin, mesh_center, VOXEL_SIZE,
                OPT_N_SOURCES, OPT_MIN_SEP_MM, OPT_NPHOTON,
                wavelength_m=808e-9
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
            # Default: 1 posterior (popliteal), 2 anterior flanking the patella.
            # +Y = anterior, -Y = posterior; Z auto-set to joint-line height per subject.
            src_configs = [
                {'name': 'Posterior',    'world_pos': [  0, -60, jl_z], 'color': 'red'  },
                {'name': 'Anterior (L)', 'world_pos': [-30,  55, jl_z], 'color': 'green'},
                {'name': 'Anterior (R)', 'world_pos': [ 30,  55, jl_z], 'color': 'blue' },
            ]
        for cfg in src_configs:
            # Aim toward the joint-line centre [0, 0, jl_z], not the geometric
            # centre [0, 0, 0], so sources point straight inward at the joint.
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
            pmcx_source_list=pmcx_source,
            source_power_mw=SOURCE_POWER_MW,
            duty_cycle=SOURCE_DUTY_CYCLE,
            opt_eff=SOURCE_OPT_EFF,
        )

        # ── Step 6: Absorption analysis ───────────────────────────────────
        results = analyze_fluence_absorption(
            fluence_combined, vol, tissues, VOXEL_SIZE,
            pmcx_source=pmcx_source  # ← pass local variable
        )

        # ── Step 7: Save subject outputs ──────────────────────────────────
        subj_dir = Path(output_dir) / melanin_condition / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 7a: Penetration depth histogram ──────────────────────────
        # Volume-weighted group mean fluence for cartilage and synovial fluid
        cart_names  = [n for n in results if 'cart' in n]
        cart_vox    = sum(results[n]['n_voxels'] for n in cart_names)
        cart_flu_mw = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                           for n in cart_names) / cart_vox) if cart_vox > 0 else 0.0

        syn_names   = [n for n in results if 'synovial' in n]
        syn_vox     = sum(results[n]['n_voxels'] for n in syn_names)
        syn_flu_mw  = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                           for n in syn_names) / syn_vox) if syn_vox > 0 else 0.0

        print("\n=== Penetration depth analysis ===")
        bin_centers, mean_flu, max_depth = analyze_penetration_depth(
            fluence_combined, vol, VOXEL_SIZE, mesh_center, origin
        )
        fig_depth = plot_depth_histogram(
            bin_centers, mean_flu, subject_id,
            cartilage_flu_mw=cart_flu_mw,
            synovial_flu_mw=syn_flu_mw,
        )
        depth_html = str(subj_dir / f"depth_histogram_{subject_id}_{melanin_condition}.html")
        fig_depth.write_html(depth_html)
        print(f"  Saved: {depth_html}")

        np.save(subj_dir / "label_volume.npy", vol)
        np.save(subj_dir / "fluence_combined.npy", fluence_combined)
        for i, flu in enumerate(fluence_list):
            np.save(subj_dir / f"fluence_src{i + 1}.npy", flu)

        return subject_id, results

    except Exception as e:
        print(f"  ERROR processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def results_to_csv(all_results, output_path="MC_Analysis_650nm.csv",
                   treatment_times_s=(300, 600, 900)):
    """
    Write absorption results to CSV with three sections per subject:
      Section 1 — Per tissue layer rows
      Section 2 — Group summary rows
      Section 3 — Fluence dose rows per group per treatment time
    """
    import csv

    # ── Group definitions ─────────────────────────────────────────────────────
    GROUPS = {
        'Bone':      lambda n: 'bone'     in n,
        'Cartilage': lambda n: 'cart'     in n,
        'Meniscus':  lambda n: 'men'      in n,
        'Synovial':  lambda n: 'synovial' in n,
        'Muscle':    lambda n: 'muscle'   in n,
        'Adipose':   lambda n: 'adipose'  in n,
        'Skin':      lambda n: 'skin'     in n,
    }

    n_sources      = 3
    power_per_src  = SOURCE_POWER_MW * SOURCE_DUTY_CYCLE * SOURCE_OPT_EFF
    total_power_mw = n_sources * power_per_src

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        for subj_id, results in all_results:

            total_absorbed_mw = sum(r['absorbed_mw'] for r in results.values())

            # ── Section 1: Per tissue layer ───────────────────────────────
            writer.writerow([])
            writer.writerow([f'Subject: {subj_id}'])
            writer.writerow([])
            writer.writerow([
                'Section', 'Subject', 'Tissue', 'Label',
                'Voxels', 'Vol (cm³)',
                'mua (mm^-1)',
                'Mean Fluence Rate (mW/cm²)',
                'Max Fluence Rate (mW/cm²)',
                'Absorbed Power (mW)',
                '% of Total Absorbed',
            ])

            sorted_results = sorted(results.items(), key=lambda kv: kv[1]['label'])
            for name, r in sorted_results:
                pct = 100 * r['absorbed_mw'] / total_absorbed_mw \
                      if total_absorbed_mw > 0 else 0
                writer.writerow([
                    'Layer',
                    subj_id,
                    name,
                    r['label'],
                    r['n_voxels'],
                    f"{r['vol_cm3']:.6f}",
                    f"{r['mua_mm']:.4f}",
                    f"{r['mean_flu']:.6e}",
                    f"{r['max_flu']:.6e}",
                    f"{r['absorbed_mw']:.6f}",
                    f"{pct:.4f}",
                ])

            # ── Section 2: Group summary ──────────────────────────────────
            writer.writerow([])
            writer.writerow([
                'Section', 'Subject', 'Group',
                'Total Voxels', 'Total Vol (cm³)',
                'Mean Fluence Rate (mW/cm²)',
                'Max Fluence Rate (mW/cm²)',
                'Absorbed Power (mW)',
                '% of Total Absorbed',
            ])

            group_data = {}
            for group, match_fn in GROUPS.items():
                group_names = [n for n in results if match_fn(n)]
                if not group_names:
                    continue

                grp_voxels   = sum(results[n]['n_voxels']    for n in group_names)
                grp_vol      = sum(results[n]['vol_cm3']      for n in group_names)
                grp_absorbed = sum(results[n]['absorbed_mw']  for n in group_names)
                grp_pct      = 100 * grp_absorbed / total_absorbed_mw \
                               if total_absorbed_mw > 0 else 0

                # Weighted mean fluence across group voxels
                grp_flu_sum  = sum(results[n]['mean_flu'] * results[n]['n_voxels']
                                   for n in group_names)
                grp_mean_flu = grp_flu_sum / grp_voxels if grp_voxels > 0 else 0
                grp_max_flu  = max(results[n]['max_flu'] for n in group_names)

                group_data[group] = {
                    'mean_flu':    grp_mean_flu,
                    'max_flu':     grp_max_flu,
                    'absorbed_mw': grp_absorbed,
                    'voxels':      grp_voxels,
                    'vol_cm3':     grp_vol,
                }

                writer.writerow([
                    'Group',
                    subj_id,
                    group,
                    grp_voxels,
                    f"{grp_vol:.6f}",
                    f"{grp_mean_flu:.6e}",
                    f"{grp_max_flu:.6e}",
                    f"{grp_absorbed:.6f}",
                    f"{grp_pct:.4f}",
                ])

            # Total row
            writer.writerow([
                'Group',
                subj_id,
                'TOTAL',
                sum(r['n_voxels'] for r in results.values()),
                f"{sum(r['vol_cm3'] for r in results.values()):.6f}",
                '',
                '',
                f"{total_absorbed_mw:.6f}",
                '100.0000',
            ])

            # Source power reference
            writer.writerow([])
            writer.writerow([
                'Power', subj_id,
                f'Sources: {n_sources}',
                f'Power per source: {power_per_src:.1f} mW (avg)',
                f'Total input: {total_power_mw:.1f} mW',
                f'Total absorbed: {total_absorbed_mw:.4f} mW',
                f'Absorption ratio: {100*total_absorbed_mw/total_power_mw:.2f}%',
            ])

            # ── Section 3: Fluence dose per group per treatment time ───────
            writer.writerow([])
            writer.writerow([
                'Section', 'Subject', 'Group',
                'Mean Fluence Rate (mW/cm²)',
            ] + [f'Dose @ {t}s (J/cm²)' for t in treatment_times_s] + [
                'Therapeutic Goal: (1-4 J/cm²)',
            ])

            for group, gdata in group_data.items():
                mean_flu = gdata['mean_flu']
                doses    = [mean_flu * 1e-3 * t for t in treatment_times_s]

                # Find treatment time range that delivers 1-4 J/cm²
                t_min = 1.0  / (mean_flu * 1e-3) if mean_flu > 0 else float('inf')
                t_max = 4.0  / (mean_flu * 1e-3) if mean_flu > 0 else float('inf')

                if t_min == float('inf'):
                    therapeutic_range = "N/A — no fluence"
                elif t_max > 3600:
                    therapeutic_range = f"> {t_min:.0f}s (>1hr for 4 J/cm²)"
                else:
                    therapeutic_range = f"{t_min:.0f}s – {t_max:.0f}s"
                '''
                writer.writerow([
                    'Dose',
                    subj_id,
                    group,
                    f"{mean_flu:.6e}",
                ] + [f"{d:.4f}" for d in doses] + [
                    therapeutic_range,
                ])
                '''
            # Blank separator between subjects
            writer.writerow([])
            writer.writerow(['-' * 40])

        # ── Section 4: Cross-subject dose summary tables ──────────────────────────
        DOSE_TIMES_S = (300, 600, 900)  # 5, 10, 15 minutes
        DOSE_TIMES_LABEL = ["5'", "10'", "15'"]

        DOSE_GROUPS = {
            'Cartilage': lambda n: 'cart' in n,
            'Muscle': lambda n: 'muscle' in n,
            'Synovial Fluid': lambda n: 'synovial' in n,
        }

        writer.writerow([])
        writer.writerow([])
        writer.writerow(['=' * 60])
        writer.writerow(['CROSS-SUBJECT DOSE SUMMARY (J/cm2)'])
        writer.writerow(['=' * 60])

        for group_name, match_fn in DOSE_GROUPS.items():

            writer.writerow([])
            writer.writerow([f'{group_name} Dose (J/cm2)'])
            writer.writerow(['Subject'] + DOSE_TIMES_LABEL)

            for subj_id, results in all_results:

                # Compute weighted mean fluence for this group
                group_names = [n for n in results if match_fn(n)]
                if not group_names:
                    writer.writerow([subj_id] + ['N/A'] * len(DOSE_TIMES_S))
                    continue

                total_voxels = sum(results[n]['n_voxels'] for n in group_names)
                if total_voxels == 0:
                    writer.writerow([subj_id] + ['N/A'] * len(DOSE_TIMES_S))
                    continue

                weighted_mean_flu = sum(
                    results[n]['mean_flu'] * results[n]['n_voxels']
                    for n in group_names
                ) / total_voxels

                # Compute dose at each treatment time
                # dose (J/cm2) = fluence_rate (mW/cm2) * time (s) * 1e-3 (W/mW)
                doses = [f"{weighted_mean_flu * 1e-3 * t:.4f}"
                         for t in DOSE_TIMES_S]

                writer.writerow([subj_id] + doses)

            # Group mean across all subjects
            writer.writerow([])
            all_means = []
            for subj_id, results in all_results:
                group_names = [n for n in results if match_fn(n)]
                total_voxels = sum(results[n]['n_voxels'] for n in group_names)
                if total_voxels > 0:
                    wm = sum(results[n]['mean_flu'] * results[n]['n_voxels']
                             for n in group_names) / total_voxels
                    all_means.append(wm)

            if all_means:
                grand_mean = np.mean(all_means)
                grand_std = np.std(all_means)
                mean_doses = [f"{grand_mean * 1e-3 * t:.4f}" for t in DOSE_TIMES_S]
                std_doses = [f"{grand_std * 1e-3 * t:.4f}" for t in DOSE_TIMES_S]

                writer.writerow(['Mean'] + mean_doses)
                writer.writerow(['StDev'] + std_doses)
                writer.writerow(['CV%'] + [
                    f"{100 * grand_std / grand_mean:.1f}%" if grand_mean > 0 else "N/A"
                    for _ in DOSE_TIMES_S
                ])

            writer.writerow([])

    print(f"\nCSV written: {output_path}  ({len(all_results)} subjects)")

VOXEL_SIZE = 1.0               # mm per voxel

# Physical grid dimensions in mm (from bounding box + padding)
GRID_DIMS_MM = (150, 140, 285)   # x, y, z in mm — edit these, not VOXEL_RES

# Compute VOXEL_RES automatically from physical size and voxel size
VOXEL_RES = tuple(int(round(d / VOXEL_SIZE)) for d in GRID_DIMS_MM)

FLUENCE_OUTPUT = None          # None to run pmcx, or True to load saved files
AUTO_ORIENT    = True          # auto-detect and correct Z-axis inversion (OKS002-type)

# Soft-tissue wrapping layer thicknesses (mm).
# Anterior knee: patella is subcutaneous, so muscle is absent — 6-8 mm is more
# realistic than 12 mm.  Posterior fossa: 15-25 mm would be accurate but makes
# sources harder to place.  These uniform values are a compromise; reducing them
# raises cartilage fluence significantly because fewer photons are absorbed en route.
MUSCLE_THICK_MM  = 8   # concentric muscle shell around bone/cartilage assembly
ADIPOSE_THICK_MM =  4   # subcutaneous fat layer outside muscle
SKIN_THICK_MM    =  2   # dermis layer (epidermis relabelled separately)

# Source power parameters — single definition used by run_pmcx, analyze_fluence_absorption,
# and results_to_csv so that all three are guaranteed consistent.
SOURCE_POWER_MW   = 50     # peak power per source (mW)
SOURCE_DUTY_CYCLE = 0.75   # modulation duty cycle
SOURCE_OPT_EFF    = 0.85   # optical coupling efficiency
# Average power delivered per source = SOURCE_POWER_MW × SOURCE_DUTY_CYCLE × SOURCE_OPT_EFF

# ─────────────────────────────────────────────────────────────────────────────
# 2. STL → VOXEL LABEL VOLUME
# ─────────────────────────────────────────────────────────────────────────────

def stl_to_voxels(mesh_path, label, origin, spacing, shape, z_flip=False):
    """Ray-cast a closed STL mesh into a voxel volume."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if not mesh.is_watertight:
        print(f"  ⚠  {mesh_path} is not watertight — attempting repair")
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
    if z_flip:
        mesh.vertices[:, 2] *= -1

    nx, ny, nz = shape
    vol = np.zeros(shape, dtype=np.uint8)

    xs = origin[0] + (np.arange(nx) + 0.5) * spacing
    ys = origin[1] + (np.arange(ny) + 0.5) * spacing

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            ray_origin = np.array([[x, y, origin[2] - spacing]])
            ray_dir    = np.array([[0.0, 0.0, 1.0]])
            locs, _, _ = mesh.ray.intersects_location(
                ray_origins=ray_origin, ray_directions=ray_dir
            )
            if len(locs) == 0:
                continue
            hit_zs = np.sort(locs[:, 2])
            for k in range(0, len(hit_zs) - 1, 2):
                z0, z1 = hit_zs[k], hit_zs[k + 1]
                iz0 = max(0, int(np.floor((z0 - origin[2]) / spacing)))
                iz1 = min(nz - 1, int(np.ceil((z1 - origin[2]) / spacing)))
                vol[ix, iy, iz0:iz1 + 1] = label
    return vol


# ─────────────────────────────────────────────────────────────────────────────
# 3. MERGE ALL STL FILES INTO LABEL VOLUME
# ─────────────────────────────────────────────────────────────────────────────

def build_label_volume(tissues, res, spacing):
    """Merge all tissue STLs into one integer label volume.

    If AUTO_ORIENT is True, detects Z-axis inversion by comparing femur vs tibia
    mean Z positions. In a standard orientation femur sits superior (higher Z) to
    tibia; if inverted the coordinates are negated before voxelization.
    """
    all_verts       = []
    femur_z_mean    = None
    tibia_z_mean    = None

    for name, (path, label, _) in tissues.items():
        if path is not None:
            m = trimesh.load(path, force="mesh")
            all_verts.append(m.vertices)
            if 'femur-bone' in name:
                femur_z_mean = m.vertices[:, 2].mean()
            elif 'tibia-bone' in name:
                tibia_z_mean = m.vertices[:, 2].mean()

    # Detect and flag Z-axis inversion
    z_flip = False
    if AUTO_ORIENT and femur_z_mean is not None and tibia_z_mean is not None:
        if femur_z_mean < tibia_z_mean:
            z_flip = True
            print(f"  [ORIENT] Z-axis inverted detected: femur_z={femur_z_mean:.1f} mm < "
                  f"tibia_z={tibia_z_mean:.1f} mm — applying Z correction")
            all_verts = [v * np.array([1.0, 1.0, -1.0]) for v in all_verts]

    verts       = np.vstack(all_verts)
    mn          = verts.min(axis=0)
    mx          = verts.max(axis=0)
    mesh_center = (mn + mx) / 2.0
    grid_half   = np.array(res) * spacing / 2.0
    origin      = mesh_center - grid_half
    mesh_dims   = mx - mn

    vol = np.zeros(res, dtype=np.uint8)
    for name, (path, label, _) in tissues.items():
        if path is not None:
            print(f"  Voxelizing {name} (label={label})...")
            layer = stl_to_voxels(path, label, origin, spacing, res, z_flip=z_flip)
            vol[layer > 0] = layer[layer > 0]

    return vol, origin, mesh_center


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADD WRAPPING LAYERS (MUSCLE, ADIPOSE, SKIN)
# ─────────────────────────────────────────────────────────────────────────────

def add_wrapping_layers(vol, layer_configs):
    """Add concentric wrapping layers around existing tissue."""
    result      = vol.copy()
    outer_shell = result > 0

    for label, thickness_vox in layer_configs:
        dilated   = binary_dilation(outer_shell, iterations=thickness_vox)
        new_layer = dilated & ~outer_shell
        result[new_layer & (result == 0)] = label
        outer_shell = dilated

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4b. ADD EPIDERMIS LAYER
# ─────────────────────────────────────────────────────────────────────────────

def add_epidermis_layer(vol, skin_label, epidermis_label):
    """
    Relabel the outermost voxels of the skin layer as epidermis (label 15).
    Physically represents the 0.2 mm stratum corneum / epidermis; at 1 mm/voxel
    resolution this maps to 1 voxel. Optical properties are set per
    MELANIN_CONDITIONS to model fair, olive, and dark skin types.
    """
    skin_mask  = vol == skin_label
    inner_skin = binary_erosion(skin_mask, iterations=1)
    epi_mask   = skin_mask & ~inner_skin
    result     = vol.copy()
    result[epi_mask] = epidermis_label
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. ADD SYNOVIAL FLUID
# ─────────────────────────────────────────────────────────────────────────────

def add_synovial_fluid(vol, cartilage_labels, bone_labels, fluid_label, dilation_vox):
    """Fill joint space between cartilage surfaces with synovial fluid."""
    cartilage_mask = np.isin(vol, cartilage_labels)
    bone_mask      = np.isin(vol, bone_labels)

    if cartilage_mask.sum() == 0:
        print("  Warning: no cartilage voxels found")
        return vol

    dilated_cart     = binary_dilation(cartilage_mask, iterations=dilation_vox)
    INNER_FILL_LABELS = set(cartilage_labels) | set(bone_labels)

    fluid_mask = (
        dilated_cart
        & ~cartilage_mask
        & ~bone_mask
        & ~np.isin(vol, list(INNER_FILL_LABELS))
    )
    result = vol.copy()
    result[fluid_mask] = fluid_label
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLACE SOURCES ON TISSUE SURFACE
# ─────────────────────────────────────────────────────────────────────────────

def find_valid_source_positions(vol, origin, spacing, mesh_center):
    """
    Print the tissue surface extent to help choose valid world_pos values.
    """
    tissue_coords = np.argwhere(vol > 0)

    # Convert all tissue voxels to centered world coordinates
    world_coords = origin + (tissue_coords + 0.5) * spacing - mesh_center

    print("\nTissue surface extent in centered world coordinates (mm):")
    print(f"  X: {world_coords[:,0].min():.1f} to {world_coords[:,0].max():.1f}")
    print(f"  Y: {world_coords[:,1].min():.1f} to {world_coords[:,1].max():.1f}")
    print(f"  Z: {world_coords[:,2].min():.1f} to {world_coords[:,2].max():.1f}")

    # Find skin surface voxels (label 13) specifically
    skin_coords  = np.argwhere(vol == 13)
    skin_world   = origin + (skin_coords + 0.5) * spacing - mesh_center

    print(f"\nSkin surface (label 13) extent:")
    print(f"  X: {skin_world[:,0].min():.1f} to {skin_world[:,0].max():.1f}")
    print(f"  Y: {skin_world[:,1].min():.1f} to {skin_world[:,1].max():.1f}")
    print(f"  Z: {skin_world[:,2].min():.1f} to {skin_world[:,2].max():.1f}")

    # Find posterior surface (max Y) at different X positions
    print(f"\nPosterior skin surface (max Y) at Z=0 slice:")
    z_mid   = int(vol.shape[2] // 2)
    z_range = 5
    for x_vox in range(0, vol.shape[0], 10):
        col = vol[x_vox, :, max(0,z_mid-z_range):min(vol.shape[2],z_mid+z_range)]
        skin_rows = np.argwhere(col == 13)
        if len(skin_rows) > 0:
            max_y_vox   = skin_rows[:,0].max()
            x_world     = origin[0] + (x_vox + 0.5) * spacing - mesh_center[0]
            y_world_max = origin[1] + (max_y_vox + 0.5) * spacing - mesh_center[1]
            print(f"  x={x_world:6.1f}mm → posterior skin at y={y_world_max:.1f}mm")

    # Find anterior surface (min Y) at different X positions
    print(f"\nAnterior skin surface (min Y) at Z=0 slice:")
    for x_vox in range(0, vol.shape[0], 10):
        col = vol[x_vox, :, max(0,z_mid-z_range):min(vol.shape[2],z_mid+z_range)]
        skin_rows = np.argwhere(col == 13)
        if len(skin_rows) > 0:
            min_y_vox   = skin_rows[:,0].min()
            x_world     = origin[0] + (x_vox + 0.5) * spacing - mesh_center[0]
            y_world_min = origin[1] + (min_y_vox + 0.5) * spacing - mesh_center[1]
            print(f"  x={x_world:6.1f}mm → anterior skin at y={y_world_min:.1f}mm")


def find_surface_source_positions(vol, origin, spacing, mesh_center, src_configs):
    """Place sources just inside the tissue surface."""
    sources       = []
    tissue_coords = np.argwhere(vol > 0)

    for cfg in src_configs:
        intended_world = np.array(cfg['world_pos'])
        intended_vox   = (intended_world + mesh_center - origin) / spacing

        distances = np.linalg.norm(tissue_coords - intended_vox, axis=1)
        nearest   = tissue_coords[distances.argmin()]

        # In find_surface_source_positions, after finding nearest
        print(f"  '{cfg['name']}': intended_vox={intended_vox}, "
              f"nearest={nearest}, "
              f"distance={distances.min():.1f} voxels, "
              f"label={vol[nearest[0], nearest[1], nearest[2]]}")

        srcdir = np.array(cfg['srcdir'], dtype=float)
        srcdir = srcdir / np.linalg.norm(srcdir)
        srcpos = nearest.astype(float).copy()

        # Walk inward until inside tissue
        for step in range(1, 21):
            sp          = [int(round(x)) for x in srcpos]
            sp_clipped  = [np.clip(sp[i], 0, vol.shape[i]-1) for i in range(3)]
            label_at_src = vol[sp_clipped[0], sp_clipped[1], sp_clipped[2]]
            if label_at_src > 0:
                break
            srcpos = nearest.astype(float) + srcdir * step

        sources.append({
            'srcpos': srcpos.tolist(),
            'srcdir': srcdir.tolist(),
            'color':  cfg['color'],
            'name':   cfg['name'],
        })

    return sources


# ─────────────────────────────────────────────────────────────────────────────
# 6b. JOINT-LINE Z DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_joint_line_z(vol, tissues, origin, spacing, mesh_center):
    """
    Return the world-Z offset of the joint line from mesh_center, using the
    same sign convention as world_pos[2] in src_configs (positive = superior).

    Method: find the Z slice that contains the most cartilage + synovial voxels.
    That slice is the densest cross-section of the joint space and is the optimal
    height at which to position sources — placing all sources at this Z maximises
    the photon fluence delivered to the target tissues by minimising the oblique
    path length from source to joint.

    Returns 0.0 if no cartilage or synovial voxels are found (safe fallback).
    """
    cart_labels = [t[1] for name, t in tissues.items() if 'cart'     in name]
    syn_labels  = [t[1] for name, t in tissues.items() if 'synovial' in name]
    target_mask = np.isin(vol, cart_labels + syn_labels)

    if target_mask.sum() == 0:
        print("  [JLINE] No cartilage/synovial voxels — using Z=0 (geometric centre)")
        return 0.0

    counts_per_z = target_mask.sum(axis=(0, 1))   # sum over X,Y per Z slice
    iz_peak      = int(np.argmax(counts_per_z))
    world_z      = origin[2] + (iz_peak + 0.5) * spacing
    z_offset     = world_z - mesh_center[2]

    print(f"  [JLINE] Joint-line Z slice: {iz_peak}  "
          f"world_z={world_z:.1f} mm  offset from centre={z_offset:+.1f} mm  "
          f"({int(counts_per_z[iz_peak])} target voxels)")
    return z_offset


# ─────────────────────────────────────────────────────────────────────────────
# 6c. RECIPROCITY-BASED SOURCE POSITION OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

def optimize_source_positions_reciprocity(vol, tissues, origin, mesh_center,
                                           spacing, n_sources, min_sep_mm,
                                           n_photon, wavelength_m=808e-9):
    """
    Find optimal external source positions using the optical reciprocity theorem.

    An isotropic virtual source is placed at the centroid of the combined
    cartilage + synovial fluid volume and a reduced-photon MC simulation is run.
    By reciprocity, the epidermis voxels that receive the most photons in this
    reverse simulation are exactly the surface positions from which an external
    source would deliver the most photons to the joint.

    A greedy non-maximum suppression (NMS) step then picks n_sources positions
    that are mutually separated by at least min_sep_mm, preventing the optimizer
    from placing all sources on the same hotspot.

    Returns: list of [x, y, z] world positions in centered coordinates
             (same convention as world_pos in src_configs).
    """
    cart_labels   = [t[1] for name, t in tissues.items() if 'cart'     in name]
    syn_labels    = [t[1] for name, t in tissues.items() if 'synovial' in name]
    target_mask   = np.isin(vol, cart_labels + syn_labels)

    if target_mask.sum() == 0:
        print("  [OPT] No cartilage/synovial voxels found — skipping optimisation")
        return None

    centroid_vox = np.argwhere(target_mask).mean(axis=0)
    print(f"  [OPT] Joint centroid (vox): {centroid_vox.round(1)}")

    # Build optical property table
    max_label  = max(t[1] for t in tissues.values())
    prop_table = [[0, 0, 1, 1]] * (max_label + 1)
    for name, (path, label, props) in tissues.items():
        prop_table[label] = props

    cfg_opt = {
        "nphoton":    n_photon,
        "srctype":    'isotropic',      # radiate equally in all directions
        "srcpos":     centroid_vox.tolist(),
        "srcdir":     [0.0, 0.0, 1.0], # ignored for isotropic
        "vol":        vol.astype(np.uint8),
        "prop":       prop_table,
        "tstart":     0,
        "tend":       1e-9,
        "tstep":      1e-9,
        "unitinmm":   spacing,
        "autopilot":  1,
        "gpuid":      1,
        "issavedet":  0,
        "outputtype": "fluence",
        "normalize":  1,
    }

    print(f"  [OPT] Reciprocity MC: {int(n_photon):.2e} photons (fast scan)...")
    res     = pmcx.run(cfg_opt)
    flu_map = res['flux'].squeeze()

    # Sample fluence on the epidermis surface only
    epi_coords = np.argwhere(vol == EPIDERMIS_LABEL)
    epi_flu    = flu_map[epi_coords[:, 0], epi_coords[:, 1], epi_coords[:, 2]]

    # Sort descending by fluence
    sort_idx   = np.argsort(epi_flu)[::-1]
    epi_coords = epi_coords[sort_idx]
    epi_flu    = epi_flu[sort_idx]
    peak_flu   = epi_flu[0]

    # Greedy NMS: pick the top-fluence voxel, suppress its neighbourhood, repeat
    min_sep_vox    = min_sep_mm / spacing
    selected_world = []
    active         = np.ones(len(epi_flu), dtype=bool)

    print(f"  [OPT] Optimal positions (min separation = {min_sep_mm:.0f} mm):")
    for i in range(n_sources):
        live = np.where(active)[0]
        if len(live) == 0:
            print(f"    Warning: only {i} positions found before candidates exhausted")
            break

        best_vox = epi_coords[live[0]]

        # Suppress all epidermis voxels within min_sep radius
        dists  = np.linalg.norm(epi_coords - best_vox, axis=1)
        active[dists < min_sep_vox] = False

        # Convert best voxel → centered world coordinates
        world_abs = origin + (best_vox.astype(float) + 0.5) * spacing
        world_cen = (world_abs - mesh_center).tolist()
        selected_world.append(world_cen)

        flu_val = flu_map[best_vox[0], best_vox[1], best_vox[2]]
        print(f"    Src {i+1}: world=[{world_cen[0]:+.1f}, {world_cen[1]:+.1f}, "
              f"{world_cen[2]:+.1f}] mm  "
              f"(rel_flu={flu_val/peak_flu:.3f}, "
              f"Y={'ant' if world_cen[1]>0 else 'post'})")

    return selected_world


# ─────────────────────────────────────────────────────────────────────────────
# 7. PMCX SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_pmcx(vol, tissues, src_cfg, pmcx_source_list,
             source_power_mw=50, wavelength_m=808e-9,
             modulation_hz=40, duty_cycle=0.75, opt_eff=0.85):

    """Run pmcx simulation and return fluence in mW/cm²."""

    h            = 6.626e-34
    c            = 3e8
    E_photon     = h * c / wavelength_m
    power_avg_W  = (source_power_mw * 1e-3) * duty_cycle * opt_eff
    Q_avg_per_s  = power_avg_W / E_photon

    # fluence (mm⁻²/ph) × Q (ph/s) × E_photon (J/ph) × 100 (mm²/cm²) × 1000 (mW/W)
    scale = Q_avg_per_s * E_photon * 100.0 * 1e3

    print(f"  Average power:  {power_avg_W*1e3:.2f} mW")

    max_label  = max(t[1] for t in tissues.values())
    prop_table = [[0, 0, 1, 1]] * (max_label + 1)
    for name, (path, label, opts) in tissues.items():
        prop_table[label] = opts

    cone_angle     = 20
    global half_angle_rad
    half_angle_rad = np.deg2rad(cone_angle / 2)

    cfg = {
        "nphoton":    1e7,
        "srctype":    'cone',
        "srcparam1":  [half_angle_rad, 0, 0, 0],
        "vol":        vol.astype(np.uint8),
        "prop":       prop_table,
        "tstart":     0,
        "tend":       1e-9,
        "tstep":      1e-9,
        "unitinmm":   VOXEL_SIZE,
        "autopilot":  1,
        "gpuid":      1,
        "issavedet":  0,
        "outputtype": "fluence",
        "normalize":  1,
    }
    cfg.update(src_cfg)

    individual_fluences = []
    combined_flux       = None

    for i, src in enumerate(pmcx_source_list):
        cfg['srcpos'] = src['srcpos']
        cfg['srcdir'] = src['srcdir']
        res           = pmcx.run(cfg)
        flux_mwcm2    = res['flux'].squeeze() * scale

        individual_fluences.append(flux_mwcm2)
        np.save(f"fluence_src{i+1}.npy", flux_mwcm2)

        combined_flux = flux_mwcm2.copy() if combined_flux is None \
                        else combined_flux + flux_mwcm2

    np.save("fluence_combined.npy", combined_flux)

    nonzero = combined_flux[combined_flux > 0]

    return combined_flux, individual_fluences


# ─────────────────────────────────────────────────────────────────────────────
# 8. TISSUE COLORS
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
    11: "rgba(180,60,60,0.4)",     # muscle
    12: "rgba(255,220,150,0.4)",   # adipose
    13: "rgba(210,180,140,0.30)",   # skin
    14: "rgba(173,216,230,0.5)",   # synovial
    15: "rgba(255,228,196,0.15)",  # epidermis
}


# ─────────────────────────────────────────────────────────────────────────────
# 9. COORDINATE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def make_coord_arrays(shape, origin, spacing, center=None):
    """Build world coordinate meshgrid, optionally centered at origin."""
    nx, ny, nz = shape
    x = origin[0] + (np.arange(nx) + 0.5) * spacing
    y = origin[1] + (np.arange(ny) + 0.5) * spacing
    z = origin[2] + (np.arange(nz) + 0.5) * spacing
    if center is not None:
        x -= center[0]
        y -= center[1]
        z -= center[2]
    return np.meshgrid(x, y, z, indexing="ij")


def voxel_to_centered_world(vox_pos, origin, spacing, mesh_center):
    """Convert voxel index to centered world coordinates matching make_coord_arrays."""
    world    = origin + (np.array(vox_pos) + 0.5) * spacing
    centered = world - mesh_center
    return centered


# ─────────────────────────────────────────────────────────────────────────────
# 10. SOURCE MARKER TRACES
# ─────────────────────────────────────────────────────────────────────────────

def add_source_traces(fig, origin, mesh_center, arrow_length=20):
    """Add source position markers and direction arrows to figure."""
    for src in PMCX_SOURCE_PLUS:
        pos       = voxel_to_centered_world(src['srcpos'], origin, VOXEL_SIZE, mesh_center)
        direction = np.array(src['srcdir'], dtype=float)
        direction /= np.linalg.norm(direction)
        end       = pos + direction * arrow_length

        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            marker=dict(size=8, color=src['color'], symbol='diamond'),
            name=src['name'],
        ))
        fig.add_trace(go.Scatter3d(
            x=[pos[0], end[0]], y=[pos[1], end[1]], z=[pos[2], end[2]],
            mode='lines',
            line=dict(color=src['color'], width=4),
            name=f"{src['name']} direction",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter3d(
            x=[end[0]], y=[end[1]], z=[end[2]],
            mode='markers',
            marker=dict(size=5, color=src['color'], symbol='circle'),
            showlegend=False,
        ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 11. PLOT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(vol, fluence_combined, fluence_list, all_fluences, fluence_names,
                 tissues, origin, spacing, smooth_sigma=1.0,
                 plot_stride=3, mesh_center=None):     # plot_stride sets the downsampling
    """Build Plotly figure with tissue isosurfaces and fluence overlay."""

    s         = plot_stride
    vol_p     = vol[::s, ::s, ::s]
    spacing_p = spacing * s
    X, Y, Z   = make_coord_arrays(vol_p.shape, origin, spacing_p, center=mesh_center)
    Xf, Yf, Zf = X.flatten(), Y.flatten(), Z.flatten()

    def prep_fluence(flu, vol_p, name=""):
        flu_ds = flu[::s, ::s, ::s]
        tissue_mask = vol_p > 0
        nonzero_flu = flu_ds[tissue_mask & (flu_ds > 0)]

        if len(nonzero_flu) == 0:
            print(f"  [{name}] No fluence — returning sentinel array")
            return np.full_like(flu_ds, -300.0)  # all sentinel, nothing renders

        floor_val = np.percentile(nonzero_flu, 10)
        ceil_val = np.percentile(nonzero_flu, 99)

        # Only include voxels that actually received photons
        # Do NOT clamp zero voxels up to floor_val — keep them at zero
        flu_masked = np.where(
            tissue_mask & (flu_ds > 0),
            np.clip(flu_ds, floor_val, ceil_val),
            0.0
        )

        # Log transform only nonzero voxels
        # Use a sentinel value for zero voxels that won't appear in isosurface
        log_floor = np.log10(floor_val)
        sentinel_val = log_floor - 3  # well below iso range, won't render

        with np.errstate(divide='ignore', invalid='ignore'):
            flu_log = np.where(
                flu_masked > 0,
                np.log10(np.maximum(flu_masked, 1e-300)),
                sentinel_val
            )

        if smooth_sigma > 0:
            # Only smooth within the illuminated region
            # Create a weight mask from actual nonzero voxels
            weight_mask = (flu_masked > 0).astype(float)
            weight_mask = gaussian_filter(weight_mask, sigma=smooth_sigma)

            flu_log_smoothed = gaussian_filter(flu_log, sigma=smooth_sigma)

            # Only accept smoothed values where there was actual illumination nearby
            # Threshold weight mask to avoid spreading to dark regions
            flu_log = np.where(
                weight_mask > 0.1,  # at least 10% illuminated neighbors.
                flu_log_smoothed,
                sentinel_val
            )

        # Final mask — zero out anything outside tissue
        flu_log = np.where(tissue_mask, flu_log, sentinel_val)

        return flu_log

    prepped = [prep_fluence(f, vol_p, name) for f, name in zip(all_fluences, fluence_names)]
    traces  = []

    # ── Tissue isosurfaces ────────────────────────────────────────────────────
    sorted_tissues = sorted(tissues.items(), key=lambda kv: kv[1][1])
    for name, (path, label, _) in sorted_tissues:
        label_mask   = gaussian_filter((vol_p == label).astype(float), sigma=0.8)
        color        = TISSUE_COLORS.get(label, "rgba(200,200,200,0.80)")
        smoothed_max = label_mask.max()

        if smoothed_max < 0.01:
            continue

        iso_thresh = smoothed_max * 0.4
        traces.append(go.Isosurface(
            x=Xf, y=Yf, z=Zf,
            value=label_mask.flatten(),
            isomin=iso_thresh,
            isomax=smoothed_max * 0.6,
            surface_count=1,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=name,
            opacity=0.85,
            visible=True,
            lighting=dict(ambient=0.7, diffuse=0.5, specular=0.2),
        ))

    n_tissue_traces_added = sum(1 for t in traces if isinstance(t, go.Isosurface))

    # ── Fluence isosurfaces ───────────────────────────────────────────────────
    tissue_mask = vol_p > 0
    for j, (flu_log, fname) in enumerate(zip(prepped, fluence_names)):
        valid_log = flu_log[tissue_mask]
        iso_vals  = np.linspace(
            np.percentile(valid_log, 10),
            np.percentile(valid_log, 99),
            5
        )
        traces.append(go.Isosurface(
            x=Xf, y=Yf, z=Zf,
            value=flu_log.flatten(),
            isomin=float(iso_vals[0]),
            isomax=float(iso_vals[-1]),
            surface_count=5,
            colorscale="Hot",
            showscale=(j == 0),
            colorbar=dict(
                title=dict(text="log₁₀ Fluence Rate<br>(mW/cm²)", side="right"),
                thickness=15, len=0.6,
            ),
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=fname,
            visible=(j == 0),
            opacity=0.25,
            lighting=dict(ambient=0.6, diffuse=0.6, specular=0.3, roughness=0.5),
        ))

    fig = go.Figure(data=traces)
    fig = add_source_traces(fig, origin, mesh_center, arrow_length=20)
    fig._n_tissue_traces = n_tissue_traces_added

    fig.update_layout(
        title="Multi-tissue voxel volume + pmcx fluence",
        scene=dict(
            xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
            bgcolor="#0d1117",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            camera=dict(
                eye=dict(x=-1.5, y=1.5, z=0.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 12. MONTE CARLO ANALYSIS: FLUENCE ABSORPTION
# ─────────────────────────────────────────────────────────────────────────────

def analyze_fluence_absorption(fluence, vol, tissues, voxel_size_mm, pmcx_source):
    """
    Compute total fluence and absorbed power per tissue layer.
    """
    voxel_vol_cm3 = (voxel_size_mm * 0.1) ** 3

    print("\n=== Fluence & Absorption Analysis ===")
    print(f"  Voxel size:   {voxel_size_mm} mm")
    print(f"  Voxel volume: {voxel_vol_cm3:.6f} cm³")

    results = {}
    total_absorbed_mw = 0.0

    print(f"\n  {'Tissue':<16} {'Label':>5} {'Voxels':>8} "
          f"{'Vol(cm³)':>10} {'mua(mm⁻¹)':>10} "
          f"{'Mean Flu':>12} {'Max Flu':>12} "
          f"{'Absorbed(mW)':>14} {'% Total':>8}")
    print("  " + "-" * 100)

    sorted_tissues = sorted(tissues.items(), key=lambda kv: kv[1][1])

    for name, (path, label, opts) in sorted_tissues:
        mua      = opts[0]
        mua_cm   = mua * 10.0
        mask     = vol == label
        n_voxels = mask.sum()

        if n_voxels == 0:
            continue

        flu_vals    = fluence[mask]
        mean_flu    = flu_vals.mean()
        max_flu     = flu_vals.max()
        vol_cm3     = n_voxels * voxel_vol_cm3
        absorbed_mw = (mua_cm * flu_vals * voxel_vol_cm3).sum()

        results[name] = {
            'label':       label,
            'n_voxels':    n_voxels,
            'vol_cm3':     vol_cm3,
            'mua_mm':      mua,
            'mean_flu':    mean_flu,
            'max_flu':     max_flu,
            'absorbed_mw': absorbed_mw,
        }
        total_absorbed_mw += absorbed_mw

    # Print per-tissue results
    for name, r in results.items():
        pct = 100 * r['absorbed_mw'] / total_absorbed_mw if total_absorbed_mw > 0 else 0
        print(f"  {name:<16} {r['label']:>5} {r['n_voxels']:>8} "
              f"{r['vol_cm3']:>10.4f} {r['mua_mm']:>10.4f} "
              f"{r['mean_flu']:>12.3e} {r['max_flu']:>12.3e} "
              f"{r['absorbed_mw']:>14.4f} {pct:>8.2f}%")

    print("  " + "-" * 100)
    print(f"  {'TOTAL':<16} {'':>5} {'':>8} {'':>10} {'':>10} "
          f"{'':>12} {'':>12} {total_absorbed_mw:>14.4f} {'100.00%':>8}")

    # Group summary
    print(f"\n  === Absorption by Group ===")
    groups = {
        'Bone':      [n for n in results if 'bone'     in n],
        'Cartilage': [n for n in results if 'cart'     in n],
        'Meniscus':  [n for n in results if 'men'      in n],
        'Synovial':  [n for n in results if 'synovial' in n],
        'Muscle':    [n for n in results if 'muscle'   in n],
        'Adipose':   [n for n in results if 'adipose'  in n],
        'Skin':      [n for n in results if 'skin'     in n],
    }
    for group, names in groups.items():
        group_absorbed = sum(results[n]['absorbed_mw'] for n in names if n in results)
        pct = 100 * group_absorbed / total_absorbed_mw if total_absorbed_mw > 0 else 0
        print(f"  {group:<12}: {group_absorbed:>10.4f} mW  ({pct:.2f}%)")

    # Source power reference
    print(f"\n  === Source Power Reference ===")
    n_sources      = len(pmcx_source)
    power_per_src  = SOURCE_POWER_MW * SOURCE_DUTY_CYCLE * SOURCE_OPT_EFF
    total_power_mw = n_sources * power_per_src
    print(f"  Sources:           {n_sources}")
    print(f"  Power per source:  {power_per_src:.1f} mW (avg)")
    print(f"  Total input power: {total_power_mw:.1f} mW")
    print(f"  Total absorbed:    {total_absorbed_mw:.4f} mW")
    print(f"  Absorption ratio:  {100*total_absorbed_mw/total_power_mw:.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 13. PENETRATION DEPTH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_penetration_depth(fluence, vol, voxel_size_mm, mesh_center, origin,
                               bin_width_cm=0.25):
    """
    Compute mean fluence rate in 0.25 cm depth bins from the outer skin surface
    to the geometric center of the knee (mesh_center).

    Depth is the Euclidean distance transform from the tissue/background boundary,
    scaled from voxels → cm.

    Returns
    -------
    bin_centers : ndarray   Depth at centre of each bin (cm)
    mean_flu    : ndarray   Mean fluence rate per bin (mW/cm²)
    max_depth_cm: float     Depth at mesh_center (cm)
    """
    tissue_mask = vol > 0

    # Distance from outer surface in voxels → cm
    depth_vox = distance_transform_edt(tissue_mask)
    depth_cm  = depth_vox * voxel_size_mm / 10.0

    # Depth at the geometric center of the knee structure
    center_vox = (mesh_center - origin) / voxel_size_mm
    ci = tuple(int(np.clip(round(center_vox[i]), 0, vol.shape[i] - 1))
               for i in range(3))
    max_depth_cm = float(depth_cm[ci])

    if max_depth_cm <= 0:
        print("  Warning: mesh center has zero depth — using EDT maximum")
        max_depth_cm = float(depth_cm.max())

    n_bins      = max(1, int(np.ceil(max_depth_cm / bin_width_cm)))
    bin_edges   = np.linspace(0.0, n_bins * bin_width_cm, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Only consider tissue voxels with nonzero fluence
    valid    = tissue_mask & (fluence > 0)
    depths   = depth_cm[valid]
    fluences = fluence[valid]

    mean_flu = np.zeros(n_bins)
    for i in range(n_bins):
        in_bin = (depths >= bin_edges[i]) & (depths < bin_edges[i + 1])
        if in_bin.sum() > 0:
            mean_flu[i] = fluences[in_bin].mean()

    print(f"  Depth range: 0 – {max_depth_cm:.2f} cm  |  "
          f"{n_bins} bins @ {bin_width_cm} cm each")

    return bin_centers, mean_flu, max_depth_cm


def plot_depth_histogram(bin_centers, mean_flu, subject_id, bin_width_cm=0.25,
                          treatment_times_s=(300, 600, 900),
                          cartilage_flu_mw=0.0, synovial_flu_mw=0.0):
    """
    Plotly bar chart: mean fluence rate (mW/cm²) vs penetration depth (cm).
    Y-axis is log-scaled. Reference lines mark approximate anatomical depths.
    Annotates zone integral dose (J/cm²) plus tissue-specific fluence for
    cartilage (group) and synovial fluid from the absorption analysis.
    """
    DEPTH_REFS = [
        (0.8,  'Skin/Adipose'),
        (2.0,  'Muscle'),
        (3.5,  'Joint space'),
    ]
    ZONE_LO, ZONE_HI = 2.0, 3.5   # cm — muscle to joint space reference lines

    bin_centers = np.asarray(bin_centers)
    mean_flu    = np.asarray(mean_flu)

    # ── Zone dose integration ──────────────────────────────────────────────────
    # Trapezoidal integration of fluence rate over depth zone, normalised by zone
    # width to recover units of mW/cm².  Dose [J/cm²] = F_norm × 1e-3 × t [s]
    zone_mask  = (bin_centers >= ZONE_LO) & (bin_centers <= ZONE_HI)
    zone_width = ZONE_HI - ZONE_LO          # 1.5 cm
    n_zone     = zone_mask.sum()
    if n_zone >= 2:
        zone_integral = float(np.trapz(mean_flu[zone_mask], bin_centers[zone_mask]))
    elif n_zone == 1:
        zone_integral = float(mean_flu[zone_mask][0] * bin_width_cm)
    else:
        zone_integral = 0.0
    zone_norm_mw  = zone_integral / zone_width   # mW/cm²
    dose_lines = [f"  {t // 60:.0f} min:  {zone_norm_mw * 1e-3 * t:.4f} J/cm²"
                  for t in treatment_times_s]
    # ── Tissue-specific dose lines ─────────────────────────────────────────────
    cart_dose_lines = [f"  {t // 60:.0f} min:  {cartilage_flu_mw * 1e-3 * t:.4f} J/cm²"
                       for t in treatment_times_s]
    syn_dose_lines  = [f"  {t // 60:.0f} min:  {synovial_flu_mw * 1e-3 * t:.4f} J/cm²"
                       for t in treatment_times_s]

    annot_text = (
        f"<b>Zone {ZONE_LO}–{ZONE_HI} cm  ∫F·dz / Δz</b><br>"
        f"Norm. fluence rate: {zone_norm_mw:.4f} mW/cm²<br>"
        + "<br>".join(dose_lines)
        + f"<br><br><b>Cartilage (group, vol-weighted)</b><br>"
        + f"Fluence rate: {cartilage_flu_mw:.4f} mW/cm²<br>"
        + "<br>".join(cart_dose_lines)
        + f"<br><br><b>Synovial Fluid</b><br>"
        + f"Fluence rate: {synovial_flu_mw:.4f} mW/cm²<br>"
        + "<br>".join(syn_dose_lines)
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=mean_flu,
        width=bin_width_cm * 0.85,
        marker=dict(
            color=mean_flu,
            colorscale='Hot',
            reversescale=True,
            showscale=True,
            colorbar=dict(
                title=dict(text='mW/cm²', side='right'),
                thickness=15,
                len=0.6,
            ),
        ),
        name='Mean Fluence Rate',
    ))

    max_depth = float(bin_centers[-1]) + bin_width_cm / 2 if len(bin_centers) else 5.0
    for depth, label in DEPTH_REFS:
        if depth <= max_depth:
            fig.add_shape(
                type='line',
                x0=depth, x1=depth,
                y0=0, y1=1,
                xref='x', yref='paper',
                line=dict(color='rgba(100,200,255,0.55)', width=1, dash='dash'),
            )
            fig.add_annotation(
                x=depth, y=1, xref='x', yref='paper',
                text=label, showarrow=False,
                font=dict(size=9, color='#8b949e'),
                xanchor='left', yanchor='bottom',
                xshift=3,
            )

    # ── Zone dose annotation box ───────────────────────────────────────────────
    fig.add_annotation(
        x=0.98, y=0.98,
        xref='paper', yref='paper',
        text=annot_text,
        showarrow=False,
        align='left',
        xanchor='right', yanchor='top',
        font=dict(size=10, color='#e6edf3'),
        bgcolor='rgba(22,27,34,0.85)',
        bordercolor='#30363d',
        borderwidth=1,
        borderpad=6,
    )

    fig.update_layout(
        title=dict(
            text=f'Fluence Rate vs Penetration Depth — {subject_id} (808 nm)',
            font=dict(size=14),
        ),
        xaxis=dict(
            title='Penetration Depth from Skin Surface (cm)',
            gridcolor='#30363d',
            zeroline=False,
            dtick=0.25,
        ),
        yaxis=dict(
            title='Mean Fluence Rate (mW/cm²)',
            type='log',
            gridcolor='#30363d',
            zeroline=False,
        ),
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font_color='#e6edf3',
        legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1),
        margin=dict(l=70, r=20, t=55, b=55),
        bargap=0.05,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 14. WRITE INTERACTIVE HTML
# ─────────────────────────────────────────────────────────────────────────────

def write_interactive_html(fig, tissues, output_path="fluence_overlay.html"):
    import json
    import re

    sorted_tissues = sorted(tissues.items(), key=lambda kv: kv[1][1])
    n_tissues      = fig._n_tissue_traces
    n_fluence      = len(all_fluences)
    n_sources      = len(PMCX_SOURCE_PLUS) * 3
    n_total        = len(fig.data)

    assert n_tissues + n_fluence + n_sources == n_total, \
        f"Trace count mismatch: {n_tissues}+{n_fluence}+{n_sources}" \
        f"={n_tissues+n_fluence+n_sources} != {n_total}"

    tissue_info = [
        {
            "name":            name,
            "trace_idx":       i,
            "label":           data[1],
            "default_visible": True
        }
        for i, (name, data) in enumerate(sorted_tissues)
    ]

    tissue_info_js = json.dumps(tissue_info)
    flu_names_js   = json.dumps(fluence_names)

    # Let Plotly write the complete HTML with correct serialization
    fig.write_html(
        output_path,
        include_plotlyjs='cdn',
        full_html=True,
        config={'displayModeBar': True, 'responsive': True}
    )

    with open(output_path, 'r') as f:
        html = f.read()

    # Find the plot div id Plotly generated
    div_id_match = re.search(r'<div id="([^"]+)"[^>]*class="plotly-graph-div"', html)
    plot_div_id  = div_id_match.group(1) if div_id_match else 'plot'

    controls_html = """
<style>
    #controls {
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 1000;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px;
        max-width: 220px;
        color: #e6edf3;
        font-family: Arial, sans-serif;
    }
    #controls h4 {
        margin: 0 0 8px 0;
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .tissue-row {
        display: flex;
        align-items: center;
        margin: 4px 0;
        gap: 8px;
    }
    .toggle-btn {
        width: 36px;
        height: 20px;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        position: relative;
        transition: background 0.2s;
        flex-shrink: 0;
    }
    .toggle-btn.on  { background: #238636; }
    .toggle-btn.off { background: #484f58; }
    .toggle-btn::after {
        content: '';
        position: absolute;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: white;
        top: 3px;
        transition: left 0.2s;
    }
    .toggle-btn.on::after  { left: 18px; }
    .toggle-btn.off::after { left: 3px; }
    .tissue-label {
        font-size: 11px;
        color: #e6edf3;
    }
    .btn-group {
        display: flex;
        gap: 4px;
        margin-top: 6px;
    }
    .btn-group button {
        flex: 1;
        background: #21262d;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 3px 6px;
        font-size: 10px;
        cursor: pointer;
    }
    .btn-group button:hover { background: #30363d; }
    #fluence-select {
        margin-top: 10px;
        border-top: 1px solid #30363d;
        padding-top: 8px;
    }
    #fluence-select h4 {
        margin: 0 0 6px 0;
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
    }
    #fluence-select select {
        width: 100%;
        background: #21262d;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 4px;
        font-size: 11px;
    }
</style>
<div id="controls">
    <h4>Tissues</h4>
    <div id="tissue-toggles"></div>
    <div class="btn-group">
        <button onclick="setAll(true)">All On</button>
        <button onclick="setAll(false)">All Off</button>
    </div>
    <div id="fluence-select">
        <h4>Fluence Source</h4>
        <select id="fluence-dropdown" onchange="switchFluence(this.value)">
        </select>
    </div>
</div>"""

    controls_js = ("""
<script>
    const tissueInfo   = """ + tissue_info_js + """;
    const nTissues     = """ + str(n_tissues) + """;
    const nFluence     = """ + str(n_fluence) + """;
    const nTotal       = """ + str(n_total) + """;
    const fluenceNames = """ + flu_names_js + """;
    const plotDiv      = document.getElementById('""" + plot_div_id + """');

    let visState = new Array(nTotal).fill(true);
    tissueInfo.forEach((t, i) => { visState[i] = t.default_visible; });
    for (let j = 0; j < nFluence; j++) {
        visState[nTissues + j] = (j === 0);
    }

    plotDiv.on('plotly_afterplot', function() {
        plotDiv.removeAllListeners('plotly_afterplot');
        Plotly.update(plotDiv, { visible: visState.map(v => !!v) });
    });

    const container = document.getElementById('tissue-toggles');
    tissueInfo.forEach((t, i) => {
        const row = document.createElement('div');
        row.className = 'tissue-row';
        const btn = document.createElement('button');
        btn.className = 'toggle-btn ' + (t.default_visible ? 'on' : 'off');
        btn.id = 'toggle-' + i;
        btn.onclick = () => toggleTissue(i);
        const lbl = document.createElement('span');
        lbl.className = 'tissue-label';
        lbl.textContent = t.name;
        row.appendChild(btn);
        row.appendChild(lbl);
        container.appendChild(row);
    });

    const sel = document.getElementById('fluence-dropdown');
    fluenceNames.forEach((name, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = name;
        sel.appendChild(opt);
    });

    function toggleTissue(i) {
        visState[i] = !visState[i];
        document.getElementById('toggle-' + i).className =
            'toggle-btn ' + (visState[i] ? 'on' : 'off');
        applyVisibility();
    }

    function setAll(state) {
        tissueInfo.forEach((t, i) => {
            visState[i] = state;
            document.getElementById('toggle-' + i).className =
                'toggle-btn ' + (state ? 'on' : 'off');
        });
        applyVisibility();
    }

    function switchFluence(idx) {
        const active = parseInt(idx);
        for (let j = 0; j < nFluence; j++) {
            visState[nTissues + j] = (j === active);
        }
        applyVisibility();
    }

    function applyVisibility() {
        Plotly.update(plotDiv, { visible: visState.map(v => !!v) });
    }
</script>""")

    html = html.replace('<body>', '<body>\n' + controls_html, 1)
    html = html.replace('</body>', controls_js + '\n</body>', 1)

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        f.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# 13. MELANIN COMPARISON CSV
# ─────────────────────────────────────────────────────────────────────────────

def melanin_comparison_to_csv(all_condition_results, output_path, wavelength_nm,
                               treatment_times_s=(300, 600, 900)):
    """
    Write a cross-condition melanin comparison table to CSV.

    Parameters
    ----------
    all_condition_results : dict
        {condition_name: [(subject_id, results_dict), ...]}
    output_path : str
    wavelength_nm : int
    """
    import csv

    conditions = list(all_condition_results.keys())   # ['fair', 'olive', 'dark']

    COMP_GROUPS = {
        'Cartilage':      lambda n: 'cart'     in n,
        'Synovial Fluid': lambda n: 'synovial' in n,
        'Muscle':         lambda n: 'muscle'   in n,
        'Bone':           lambda n: 'bone'     in n,
        'Skin+Epidermis': lambda n: 'skin'     in n or 'epidermis' in n,
    }

    def vol_weighted_mean(results_dict, match_fn):
        names     = [n for n in results_dict if match_fn(n)]
        total_vox = sum(results_dict[n]['n_voxels'] for n in names)
        if total_vox == 0:
            return 0.0
        return (sum(results_dict[n]['mean_flu'] * results_dict[n]['n_voxels']
                    for n in names) / total_vox)

    # Union of all subject IDs in order
    seen, all_subj = set(), []
    for cond_list in all_condition_results.values():
        for subj_id, _ in cond_list:
            if subj_id not in seen:
                seen.add(subj_id)
                all_subj.append(subj_id)

    # Build fast lookup: {condition: {subj_id: results_dict}}
    lookup = {
        cond: dict(pairs)
        for cond, pairs in all_condition_results.items()
    }

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([f'MELANIN CONDITION COMPARISON — {wavelength_nm} nm'])
        writer.writerow(['Fitzpatrick scale: fair = I-II  |  olive = III-IV  |  dark = V-VI'])
        writer.writerow(['Epidermis: 0.2 mm physical (1 voxel at 1 mm/voxel resolution)'])
        writer.writerow([])

        cond_headers = [c.capitalize() for c in conditions]

        for group_name, match_fn in COMP_GROUPS.items():
            writer.writerow([f'=== {group_name} ==='])
            writer.writerow([])

            # ── Fluence rate table ────────────────────────────────────────
            writer.writerow(['Fluence Rate (mW/cm²)'] + cond_headers)
            group_vals = {c: [] for c in conditions}

            for subj_id in all_subj:
                row = [subj_id]
                for cond in conditions:
                    if subj_id in lookup[cond]:
                        v = vol_weighted_mean(lookup[cond][subj_id], match_fn)
                        row.append(f'{v:.4f}')
                        if v > 0:
                            group_vals[cond].append(v)
                    else:
                        row.append('N/A')
                writer.writerow(row)

            # Cross-subject stats
            for label, fn in [('Mean', np.mean), ('StDev', np.std)]:
                row = [label]
                for cond in conditions:
                    vals = group_vals[cond]
                    row.append(f'{fn(vals):.4f}' if vals else 'N/A')
                writer.writerow(row)

            writer.writerow([])

            # ── Dose tables ───────────────────────────────────────────────
            for t in treatment_times_s:
                writer.writerow([f'Dose @ {t // 60:.0f} min (J/cm²)'] + cond_headers)

                dose_vals = {c: [] for c in conditions}
                for subj_id in all_subj:
                    row = [subj_id]
                    for cond in conditions:
                        if subj_id in lookup[cond]:
                            v = vol_weighted_mean(lookup[cond][subj_id], match_fn)
                            dose = v * 1e-3 * t
                            row.append(f'{dose:.4f}')
                            if v > 0:
                                dose_vals[cond].append(dose)
                        else:
                            row.append('N/A')
                    writer.writerow(row)

                for label, fn in [('Mean', np.mean), ('StDev', np.std)]:
                    row = [label]
                    for cond in conditions:
                        vals = dose_vals[cond]
                        row.append(f'{fn(vals):.4f}' if vals else 'N/A')
                    writer.writerow(row)

                writer.writerow([])

            writer.writerow([])

    print(f"\nMelanin comparison CSV written: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 14. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    start_time  = time.perf_counter()
    SUBJECT_IDS = [f"OKS{i:03d}" for i in range(1, 10) if i != 5]
    BASE_DIR    = Path(".")
    RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR  = Path(f"results_808nm_{RUN_ID}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    print(f"Processing {len(SUBJECT_IDS)} subjects × "
          f"{len(MELANIN_CONDITIONS)} melanin conditions")
    print(f"Subjects: {SUBJECT_IDS}")

    all_condition_results = {}   # {condition: [(subj_id, results), ...]}

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
            results_to_csv(cond_results, output_path=str(csv_path))
            print(f"  Completed {len(cond_results)} of {len(SUBJECT_IDS)} subjects")

    # Cross-condition comparison CSV
    melanin_comparison_to_csv(
        all_condition_results,
        output_path=str(OUTPUT_DIR / "MC_Melanin_Comparison_808nm.csv"),
        wavelength_nm=808,
    )

    end_time = time.perf_counter()
    print(f"\nTotal elapsed: {end_time - start_time:.2f} seconds")
