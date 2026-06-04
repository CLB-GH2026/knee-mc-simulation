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
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from pathlib import Path
import webbrowser
import os
import plotly.io as pio

base_dir = Path()
mesh_dir = base_dir / 'Raw_Mesh_Files'    #Pulling stl files downloaded from OpenKnee.org

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────

start_time = time.perf_counter()

def opt(mua, mus_prime, g, n):
    """Convert reduced scattering to transport scattering for pmcx."""
    return [mua, mus_prime / (1 - g), g, n]

# Map tissue name → (stl_path, label_id, optical_props [mua, mus, g, n])
# Optical properties at 750 nm (example values — replace with your own)

TISSUES = {
    "synovial":     (None, 14, opt(0.0002, 0.01,  0.90, 1.36)),
    "skin":         (None, 13, opt(0.0046, 1.22,  0.79, 1.40)),
    "adipose":      (None, 12, opt(0.0057, 1.00,  0.90, 1.44)),
    "muscle":       (None, 11, opt(0.0180, 0.55,  0.93, 1.37)),
    "pat-cart":     (mesh_dir/"patella_lig_raw.stl",         10, opt(0.0050, 1.50, 0.90, 1.37)),
    "mtc-cart":     (mesh_dir/"tibia_cartilage_med_raw.stl",  9, opt(0.0050, 1.50, 0.90, 1.37)),
    "ltc-cart":     (mesh_dir/"tibia_cartilage_lat_raw.stl",  8, opt(0.0050, 1.50, 0.90, 1.37)),
    "fc-cart":      (mesh_dir/"femur_cartilage_raw.stl",      7, opt(0.0050, 1.50, 0.90, 1.37)),
    "mm-men":       (mesh_dir/"men_med_raw.stl",              6, opt(0.0060, 1.80, 0.90, 1.37)),
    "lm-men":       (mesh_dir/"men_lat_raw.stl",              5, opt(0.0060, 1.80, 0.90, 1.37)),
    "patella-bone": (mesh_dir/"patella_raw.stl",              4, opt(0.0130, 2.50, 0.92, 1.37)),
    "fibula-bone":  (mesh_dir/"fibula_raw.stl",               3, opt(0.0130, 2.50, 0.92, 1.37)),
    "tibia-bone":   (mesh_dir/"tibia_raw.stl",                2, opt(0.0130, 2.50, 0.92, 1.37)),
    "femur-bone":   (mesh_dir/"femur_raw.stl",                1, opt(0.0130, 2.50, 0.92, 1.37)),
}

# Define sources in centered world coordinates (mm)
# Positions are relative to mesh center (0,0,0)
# Directions are computed automatically toward center in main block
SRC_WORLD_CONFIGS = [
    {
        'name':      'Source 1',
        'world_pos': [-30, 60, 0],   # adjust to your knee geometry
        'color':     'red',
    },
    {
        'name':      'Source 2',
        'world_pos': [30, 60, 0],
        'color':     'blue',
    },
    {
        'name':      'Source 3',
        'world_pos': [0, -60, 0],
        'color':     'green',
    },
]

VOXEL_RES  = (200, 200, 200)   # (nx, ny, nz) — increase for finer detail
VOXEL_SIZE = 1.0               # mm per voxel was 1.0

FLUENCE_OUTPUT = None          # set to path string to load saved fluence,
                               # or None to run pmcx now

# ─────────────────────────────────────────────────────────────────────────────
# 2. STL → VOXEL LABEL VOLUME
# ─────────────────────────────────────────────────────────────────────────────

def stl_to_voxels(mesh_path: str, label: int,
                  origin: np.ndarray, spacing: float,
                  shape: tuple) -> np.ndarray:
    """
    Ray-cast a closed STL mesh into a boolean voxel volume.
    Returns an integer array (0 outside, `label` inside).
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    if not mesh.is_watertight:
        print(f"  ⚠  {mesh_path} is not watertight — attempting repair")
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)

    nx, ny, nz = shape
    vol = np.zeros(shape, dtype=np.uint8)

    # Build world coordinates for voxel centres
    xs = origin[0] + (np.arange(nx) + 0.5) * spacing
    ys = origin[1] + (np.arange(ny) + 0.5) * spacing
    zs = origin[2] + (np.arange(nz) + 0.5) * spacing

    # Cast rays along Z for every (x, y) column
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            ray_origin  = np.array([[x, y, origin[2] - spacing]])
            ray_dir     = np.array([[0.0, 0.0, 1.0]])
            locs, idx_ray, idx_tri = mesh.ray.intersects_location(
                ray_origins=ray_origin, ray_directions=ray_dir
            )
            if len(locs) == 0:
                continue
            hit_zs = np.sort(locs[:, 2])
            # parity fill
            for k in range(0, len(hit_zs) - 1, 2):
                z0, z1 = hit_zs[k], hit_zs[k + 1]
                iz0 = max(0, int(np.floor((z0 - origin[2]) / spacing)))
                iz1 = min(nz - 1, int(np.ceil((z1 - origin[2]) / spacing)))
                vol[ix, iy, iz0:iz1 + 1] = label
    return vol


# ─────────────────────────────────────────────────────────────────────────────
# 3. MERGE ALL STL FILES INTO LABEL VOLUME
# ─────────────────────────────────────────────────────────────────────────────

def build_label_volume(tissues: dict, res: tuple, spacing: float) -> np.ndarray:
    """
    Merge all tissue STLs into one integer label volume.
    Later tissues (higher label) overwrite earlier ones where they overlap.
    """
    # Determine bounding box across all meshes
    all_verts = []
    for name, (path, label, _) in tissues.items():
        if path is not None:  # ← skip synthetic layers
            m = trimesh.load(path, force="mesh")
            all_verts.append(m.vertices)

    verts = np.vstack(all_verts)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)

    # True center of the mesh bounding box
    mesh_center = (mn + mx) / 2.0

    # Place origin so the mesh center lands at the voxel grid center
    grid_half = np.array(res) * spacing / 2.0
    origin = mesh_center - grid_half

    # override res to honour requested grid
    shape = res

    vol = np.zeros(shape, dtype=np.uint8)
    for name, (path, label, _) in tissues.items():
        if path is not None:
            print(f"  Voxelizing {name} (label={label})…")
            layer = stl_to_voxels(path, label, origin, spacing, shape)
            mask  = layer > 0
            vol[mask] = layer[mask]

    return vol, origin, mesh_center

# ─────────────────────────────────────────────────────────────────────────────
# 4. ADD LAYERS NOT VOXELIZED (SKIN, MUSCLE, ADIPOSE TISSUE) AS OUTER WRAP
# ─────────────────────────────────────────────────────────────────────────────

def add_wrapping_layers(vol, layer_configs):
    result = vol.copy()
    outer_shell = result > 0

    for label, thickness_vox in layer_configs:   # ← only 2 values
        dilated = binary_dilation(outer_shell, iterations=thickness_vox)
        new_layer = dilated & ~outer_shell
        empty_mask = result == 0
        result[new_layer & empty_mask] = label
        outer_shell = dilated

    return result

# ─────────────────────────────────────────────────────────────────────────────
# 5. ADD SYNOVIAL FLUID TO VOIDED VOLUME BETWEEN TISSUES
# ─────────────────────────────────────────────────────────────────────────────

def add_synovial_fluid(vol, cartilage_labels, bone_labels, fluid_label, dilation_vox):
    # Synovial fluid as added basically wherever there is a void in the volume
    from scipy.ndimage import binary_dilation

    cartilage_mask = np.isin(vol, cartilage_labels)
    bone_mask = np.isin(vol, bone_labels)

    if cartilage_mask.sum() == 0:
        print("  Warning: no cartilage voxels found")
        return vol

    # Dilate from each cartilage surface to bridge opposing surfaces
    dilated_cart = binary_dilation(cartilage_mask, iterations=dilation_vox)

    # Only fill voxels that are:
    # - within dilated cartilage region (i.e. in the joint gap)
    # - not already cartilage or bone
    # - currently background OR muscle that has invaded the joint space
    INNER_FILL_LABELS = set(cartilage_labels) | set(bone_labels)

    fluid_mask = (
            dilated_cart
            & ~cartilage_mask
            & ~bone_mask
            & ~np.isin(vol, list(INNER_FILL_LABELS))
    )

    print(f"  Synovial fluid voxels: {fluid_mask.sum()}")
    result = vol.copy()
    result[fluid_mask] = fluid_label
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 6. DEFINE SURFACE POSITIONS TO PLACE SOURCES (TO ENSURE RAYS LAUNCHED INTO TISSUE VOLUME)
# ─────────────────────────────────────────────────────────────────────────────

def find_surface_source_positions(vol, origin, spacing, mesh_center, src_configs):
    """
    Place sources just inside the skin surface so pmcx
    launches photons into tissue immediately.
    """
    sources = []
    tissue_coords = np.argwhere(vol > 0)

    for cfg in src_configs:
        intended_world = np.array(cfg['world_pos'])
        intended_vox = (intended_world + mesh_center - origin) / spacing

        # Find nearest tissue voxel
        distances = np.linalg.norm(tissue_coords - intended_vox, axis=1)
        nearest = tissue_coords[distances.argmin()]

        # Step ONE voxel INWARD along source direction
        # so source is just inside tissue not just outside
        srcdir = np.array(cfg['srcdir'])
        srcdir = srcdir / np.linalg.norm(srcdir)

        # Move inward by 1 voxel (into tissue)
        srcpos = nearest.astype(float) + srcdir * 1.0

        # Verify the position is inside tissue
        sp = [int(round(x)) for x in srcpos]
        sp_clipped = [np.clip(sp[i], 0, vol.shape[i] - 1) for i in range(3)]
        label_at_src = vol[sp_clipped[0], sp_clipped[1], sp_clipped[2]]

        # If still in background, walk inward until we hit tissue
        max_steps = 20
        step = 1
        while label_at_src == 0 and step < max_steps:
            srcpos = nearest.astype(float) + srcdir * (step + 1)
            sp = [int(round(x)) for x in srcpos]
            sp_clipped = [np.clip(sp[i], 0, vol.shape[i] - 1) for i in range(3)]
            label_at_src = vol[sp_clipped[0], sp_clipped[1], sp_clipped[2]]
            step += 1

        print(f"  Source '{cfg['name']}':")
        print(f"    Nearest tissue vox:  {nearest}, "
              f"label={vol[nearest[0], nearest[1], nearest[2]]}")
        print(f"    Final srcpos (vox):  {srcpos}")
        print(f"    Label at srcpos:     {label_at_src}")
        print(f"    Steps inward:        {step}")

        sources.append({
            'srcpos': srcpos.tolist(),
            'srcdir': srcdir.tolist(),
            'color': cfg['color'],
            'name': cfg['name'],
        })

    return sources


# ─────────────────────────────────────────────────────────────────────────────
# 7. PMCX SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_pmcx(vol, tissues, src_cfg,
             source_power_mw = 50,
             wavelength_m    = 808e-9,
             modulation_hz   = 40,
             duty_cycle      = 0.75):

    """
    Build pmcx config from label volume + tissue optical props, run, return fluence.
    """

    # ── Unit conversion constants ─────────────────────────────────────────────────
    MM2_TO_CM2 = 100  # 1 mm⁻² = 1e-2 cm⁻²  (1 cm² = 100 mm²)
    W_TO_MW = 1e3  # 1 W = 1000 mW

    h            = 6.626e-34
    c            = 3e8
    E_photon     = h * c / wavelength_m       # J/photon  ~2.65e-19 J at 750nm
    power_peak_W = source_power_mw * 1e-3     # W
    power_avg_W  = power_peak_W * duty_cycle  # W

    Q_avg_per_s  = power_avg_W / E_photon     # ph/s

    # Unit chain:
    # flux (mm⁻²/ph) × Q (ph/s) × E_photon (J/ph) × 100 (mm²/cm²) × 1000 (mW/W)
    scale = Q_avg_per_s * E_photon * 100.0 * 1e3

    print(f"  E_photon:            {E_photon:.3e} J")
    print(f"  Average power:       {power_avg_W*1e3:.2f} mW")
    print(f"  Q_avg:               {Q_avg_per_s:.3e} ph/s")
    print(f"  Scale factor:        {scale:.3e} mW/cm² per (mm⁻²/ph)")

    # Expected scale factor sanity check:
    # At 750nm, 37.5mW average:
    # E_photon  ≈ 2.65e-19 J
    # Q_avg     ≈ 1.42e17 ph/s
    # scale     ≈ 1.42e17 × 2.65e-19 × 100 × 1000
    #           ≈ 37.5 × 1e5
    #           ≈ 3.75e6
    # This means a voxel with flux=1e-5 mm⁻²/ph → ~37.5 mW/cm² ✓

    max_label  = max(t[1] for t in tissues.values())
    prop_table = [[0, 0, 1, 1]] * (max_label + 1)
    for name, (path, label, opts) in tissues.items():
        prop_table[label] = opts

    cone_angle = 32
    global half_angle_rad
    half_angle_rad = np.deg2rad(cone_angle / 2)

    cfg = {
        "nphoton":   1e8,
        "srctype":   'cone',
        'srcparam1': [half_angle_rad, 0, 0, 0],
        "vol":       vol.astype(np.uint8),
        "prop":      prop_table,
        "tstart":    0,
        "tend":      1e-9,
        "tstep":     1e-9,
        "unitinmm":  VOXEL_SIZE,
        "autopilot": 1,
        "gpuid":     1,
        "issavedet": 0,
        "outputtype": "fluence",  # mm⁻² per launched photon
        "normalize": 1,  # ensure normalization by nphoton
    }
    cfg.update(src_cfg)

    print("pmcx cfg:")
    for k, v in cfg.items():
        if k != 'vol':  # skip printing the whole volume
            print(f"  {k}: {v}")

    individual_fluences = []
    combined_flux = None

    for i, src in enumerate(PMCX_SOURCE):
        # Testing Code ---------------------------------
        sp = [int(round(x)) for x in src['srcpos']]
        print(f"\nSource {i + 1} at voxel {sp}:")

        # Sample a region around the source
        r = 5
        region = vol[
            max(0, sp[0] - r):min(vol.shape[0], sp[0] + r),
            max(0, sp[1] - r):min(vol.shape[1], sp[1] + r),
            max(0, sp[2] - r):min(vol.shape[2], sp[2] + r)
        ]
        print(f"  Labels in ±{r} voxel neighborhood: {np.unique(region)}")
        print(f"  Nonzero fraction: {(region > 0).mean():.2f}")

        # Find nearest tissue voxel
        tissue_coords = np.argwhere(vol > 0)
        distances = np.linalg.norm(tissue_coords - np.array(sp), axis=1)
        nearest_idx = distances.argmin()
        nearest_vox = tissue_coords[nearest_idx]
        print(f"  Nearest tissue voxel: {nearest_vox}, "
              f"distance={distances[nearest_idx]:.1f} voxels, "
              f"label={vol[nearest_vox[0], nearest_vox[1], nearest_vox[2]]}")
        # Testing code -------------------------------
        cfg['srcpos'] = src['srcpos']
        cfg['srcdir'] = src['srcdir']
        res = pmcx.run(cfg)
        flux_mwcm2 = res['flux'].squeeze() * scale

        raw_flux = res['flux'].squeeze()
        nonzero = raw_flux[raw_flux > 0]


        individual_fluences.append(flux_mwcm2)
        np.save(f"fluence_src{i+1}.npy", flux_mwcm2)

        if combined_flux is None:
            combined_flux = flux_mwcm2.copy()
        else:
            combined_flux += flux_mwcm2

    np.save("fluence_combined.npy", combined_flux)

    nonzero = combined_flux[combined_flux > 0]
    print(f"  Fluence rate max:          {combined_flux.max():.3e} mW/cm²")
    print(f"  Fluence rate min (nonzero):{nonzero.min():.3e} mW/cm²")
    print(f"  Dynamic range:             {combined_flux.max()/nonzero.min():.1e}")

    return combined_flux, individual_fluences

# ─────────────────────────────────────────────────────────────────────────────
# 8. DEFINE TISSUE PROPERTIES FOR PLOTLY VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

# Distinct colours per tissue types and opacity (0% to 100%)
TISSUE_COLORS = {
    1:  "rgba(128,128,128,1.00)",   # femur-bone    - gray
    2:  "rgba(128,128,128,1.00)",   # tibia-bone    - gray
    3:  "rgba(128,128,128,1.00)",   # fibula-bone   - gray
    4:  "rgba(128,128,128,1.00)",   # patella-bone  - gray
    5:  "rgba(0,0,255,1.00)",       # lm-men        - blue
    6:  "rgba(0,0,255,1.00)",       # mm-men        - blue
    7:  "rgba(0,0,255,1.00)",       # fc-cart       - blue
    8:  "rgba(0,0,255,1.00)",       # ltc-cart      - blue
    9:  "rgba(0,0,255,1.00)",       # mtc-cart      - blue
    10: "rgba(0,0,255,1.00)",       # pat-cart      - blue
    11: "rgba(180,60,60,0.20)",     # muscle        - red   Was 0.9 opacity
    12: "rgba(255,220,150,0.15)",   # adipose       - pale yellow    Was 0.7 opacity
    13: "rgba(210,180,140,0.10)",   # skin          - tan   # opacity was 0.60
    14: "rgba(173,216,230,0.45)",   # synovial      - light blue
}

# ─────────────────────────────────────────────────────────────────────────────
# 9. SHIFT WORLD COORDINATE SYSTEM TO (0,0,0) GRID SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def make_coord_arrays(shape, origin, spacing, center=None):
    # Shift mesh volumes into center of world coordinate grid
    nx, ny, nz = shape
    x = origin[0] + (np.arange(nx) + 0.5) * spacing
    y = origin[1] + (np.arange(ny) + 0.5) * spacing
    z = origin[2] + (np.arange(nz) + 0.5) * spacing

    # Shift so center of volume lands at (0, 0, 0)
    if center is not None:
        x -= center[0]
        y -= center[1]
        z -= center[2]

    return np.meshgrid(x, y, z, indexing="ij")

# ─────────────────────────────────────────────────────────────────────────────
# 10. CONVERT VOXEL COUNT TO CENTERED WORLD COORDINATE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def voxel_to_centered_world(vox_pos, origin, spacing, mesh_center):
    """Convert voxel index position to the same centered world coords as the plot."""
    # vox = np.array(vox_pos)
    world = origin + (np.array(vox_pos) + 0.5) * spacing
    centered = world - mesh_center
    return centered

# ─────────────────────────────────────────────────────────────────────────────
# 11. ADD MARKER AND LINE FOR SOURCE POSITION AND DIRECTION
# ─────────────────────────────────────────────────────────────────────────────

def add_source_traces(fig, origin, mesh_center, arrow_length=20):
    # Add a marker for each source position with a line direction for visual feedback
    for src in PMCX_SOURCE_PLUS:
        pos = voxel_to_centered_world(src['srcpos'], origin, VOXEL_SIZE, mesh_center)
        direction = np.array(src['srcdir'], dtype=float)
        direction /= np.linalg.norm(direction)  # normalize just in case
        end = pos + direction * arrow_length

        # Marker at source position
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            marker=dict(size=8, color=src['color'], symbol='diamond'),
            name=src['name'],
        ))

        # Line showing direction
        fig.add_trace(go.Scatter3d(
            x=[pos[0], end[0]],
            y=[pos[1], end[1]],
            z=[pos[2], end[2]],
            mode='lines',
            line=dict(color=src['color'], width=4),
            name=f"{src['name']} direction",
            showlegend=True,
        ))

        # Arrowhead marker at tip
        fig.add_trace(go.Scatter3d(
            x=[end[0]], y=[end[1]], z=[end[2]],
            mode='markers',
            marker=dict(size=5, color=src['color'], symbol='circle'),
            showlegend=False,
        ))

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 12. CONVERT NUMPY ARRAYS TO PYTHON LISTS FOR HTML PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_json_safe(fig):
    """
    Serialize figure to JSON ensuring all numpy arrays are
    fully converted to Python lists before serialization.
    """
    import json
    import numpy as np

    fig_dict = fig.to_dict()

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj

    fig_dict_converted = convert(fig_dict)

    # Verify before serializing
    first = fig_dict_converted['data'][0]
    print(f"  Safe serialization check:")
    print(f"    value present: {'value' in first}")
    print(f"    value length:  {len(first.get('value', []))}")
    print(f"    x length:      {len(first.get('x', []))}")
    print(f"    isomin:        {first.get('isomin')}")
    print(f"    isomax:        {first.get('isomax')}")

    return json.dumps(fig_dict_converted)



# ─────────────────────────────────────────────────────────────────────────────
# 12. WRITE INTERACTIVE VISUAL DISPLAY HTML FILE THAT WILL BE CALLED FOR WINDOW
# ─────────────────────────────────────────────────────────────────────────────

def write_interactive_html(fig, tissues, output_path="fluence_overlay.html"):
    import json
    import re

    OUTER_LABELS = {
        t[1] for name, t in tissues.items()
        if name in ("muscle", "adipose", "skin")
    }

    sorted_tissues = sorted(tissues.items(), key=lambda kv: kv[1][1])
    n_tissues  = fig._n_tissue_traces
    n_fluence  = len(all_fluences)
    n_sources  = len(PMCX_SOURCE_PLUS) * 3
    n_total    = len(fig.data)

    print(f"  HTML trace accounting:")
    print(f"    tissue traces:  {n_tissues}")
    print(f"    fluence traces: {n_fluence}")
    print(f"    source traces:  {n_sources}")
    print(f"    total expected: {n_tissues + n_fluence + n_sources}")
    print(f"    actual in fig:  {n_total}")

    assert n_tissues + n_fluence + n_sources == n_total, \
        f"Trace count mismatch: {n_tissues}+{n_fluence}+{n_sources}" \
        f"={n_tissues + n_fluence + n_sources} != {n_total}"

    tissue_info = [
        {
            "name":            name,
            "trace_idx":       i,
            "label":           data[1],
            "default_visible": data[1] not in OUTER_LABELS
        }
        for i, (name, data) in enumerate(sorted_tissues)
    ]

    tissue_info_js = json.dumps(tissue_info)
    flu_names_js   = json.dumps(fluence_names)

    # ── Step 1: Let Plotly write the complete HTML with correct serialization ──
    fig.write_html(
        output_path,
        include_plotlyjs='cdn',
        full_html=True,
        config={'displayModeBar': True, 'responsive': True}
    )

    # ── Step 2: Read it back ──────────────────────────────────────────────────
    with open(output_path, 'r') as f:
        html = f.read()

    # ── Step 3: Find the plot div id Plotly generated ─────────────────────────
    div_id_match = re.search(
        r'<div id="([^"]+)"[^>]*class="plotly-graph-div"', html
    )
    if div_id_match:
        plot_div_id = div_id_match.group(1)
        print(f"  Found plot div id: {plot_div_id}")
    else:
        plot_div_id = 'plot'
        print(f"  Warning: could not find plot div id, using default")

    # ── Step 4: Build controls div ────────────────────────────────────────────
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

    # ── Step 5: Build controls JS ─────────────────────────────────────────────
    controls_js = ("""
<script>
    const tissueInfo   = """ + tissue_info_js + """;
    const nTissues     = """ + str(n_tissues) + """;
    const nFluence     = """ + str(n_fluence) + """;
    const nTotal       = """ + str(n_total) + """;
    const fluenceNames = """ + flu_names_js + """;
    const plotDiv      = document.getElementById('""" + plot_div_id + """');

    let visState = new Array(nTotal).fill(true);
    tissueInfo.forEach((t, i) => {
        visState[i] = t.default_visible;
    });
    for (let j = 0; j < nFluence; j++) {
        visState[nTissues + j] = (j === 0);
    }

    plotDiv.on('plotly_afterplot', function() {
        plotDiv.removeAllListeners('plotly_afterplot');
        console.log('Plot ready, applying visibility');
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

    # ── Step 6: Inject into Plotly HTML ───────────────────────────────────────
    html = html.replace('<body>', '<body>\n' + controls_html, 1)
    html = html.replace('</body>', controls_js + '\n</body>', 1)

    # ── Step 7: Write final file ──────────────────────────────────────────────
    print(f"  HTML length: {len(html)/1e6:.1f} MB")
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"  Written {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. DEFINE HOW PLOTLY WILL PLOT OVERLAYED TISSUE AND FLUENCE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(vol, fluence_combined, fluence_list, all_fluences, fluence_names,
                 tissues, origin, spacing, smooth_sigma=1.0,
                 plot_stride=6, mesh_center=None):
    """
    Plotly figure with:
      • One semi-transparent Isosurface per tissue label
      • Fluence Isosurface coloured by log10(fluence), hot colorscale
    """

    s = plot_stride
    vol_p = vol[::s, ::s, ::s]
    spacing_p = spacing * s     # voxel size grows with stride

    X, Y, Z = make_coord_arrays(vol_p.shape, origin, spacing_p, center=mesh_center)
    Xf, Yf, Zf = X.flatten(), Y.flatten(), Z.flatten()

    # Build all fluence variants: combined + each individual

    def prep_fluence(flu, vol_p):
        flu_ds = flu[::s, ::s, ::s]
        tissue_mask = vol_p > 0
        tissue_flu = flu_ds[tissue_mask]
        nonzero_flu = tissue_flu[tissue_flu > 0]

        if len(nonzero_flu) == 0:
            print("  Warning: no valid fluence voxels found")
            return np.zeros_like(flu_ds)

        # Use percentile bounds to exclude noise floor and outliers
        floor_val = np.percentile(nonzero_flu, 10)  # ~5e-6 mw/cm²
        ceil_val = np.percentile(nonzero_flu, 99)  # ~8 mw/cm²

        print(f"  Nonzero voxels:        {len(nonzero_flu)} of {tissue_mask.sum()}")
        print(f"  Floor (10th pct):      {floor_val:.3e} mW/cm²  "
              f"(log10={np.log10(floor_val):.1f})")
        print(f"  Ceiling (99th pct):    {ceil_val:.3e} mW/cm²  "
              f"(log10={np.log10(ceil_val):.1f})")
        print(f"  Display dynamic range: {ceil_val / floor_val:.1e}  "
              f"({np.log10(ceil_val / floor_val):.1f} decades)")

        # Clamp fluence to floor/ceiling range
        flu_clamped = np.where(
            tissue_mask & (flu_ds >= floor_val),
            np.clip(flu_ds, floor_val, ceil_val),
            floor_val
        )

        # Log transform
        flu_log = np.log10(flu_clamped)

        if smooth_sigma > 0:
            flu_log = gaussian_filter(flu_log, sigma=smooth_sigma)

        # Re-apply tissue mask after smoothing to remove boundary bleed
        flu_log = np.where(tissue_mask, flu_log, np.log10(floor_val) - 1)

        print(f"  Final log10 range:     {flu_log[tissue_mask].min():.2f} "
              f"to {flu_log[tissue_mask].max():.2f}")
        print(f"  Final dynamic range:   "
              f"{10 ** (flu_log[tissue_mask].max() - flu_log[tissue_mask].min()):.1e}")

        return flu_log

    prepped = [prep_fluence(f, vol_p) for f in all_fluences]

    traces = []

    #  n_tissues = len(tissues)    Removed to be replaced so traces only counted if actually added
    n_fluence_variants = len(all_fluences)

    # ── Tissue isosurfaces ────────────────────────────────────────────────────
    sorted_tissues = sorted(tissues.items(), key=lambda kv: kv[1][1])
    for name, (path, label, _) in sorted_tissues:
        label_mask = (vol_p == label).astype(float)  # renamed
        label_mask = gaussian_filter(label_mask, sigma=0.8)  # renamed
        color = TISSUE_COLORS.get(label, "rgba(200,200,200,0.80)")
        smoothed_max = label_mask.max()  # renamed

        print(f"  {name}: label={label}, smoothed_max={smoothed_max:.4f}, "
              f"voxels={(vol_p == label).sum()}")

        if smoothed_max < 0.01:
            print(f"    SKIPPING {name}")
            continue

        iso_thresh = smoothed_max * 0.4

        traces.append(go.Isosurface(
            x=Xf, y=Yf, z=Zf,
            value=label_mask.flatten(),  # renamed
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

    # ── overall tissue mask for fluence clamping ─────────────────────────────
    # Now tissue_mask is unambiguously the full volume tissue mask
    tissue_mask = vol_p > 0

    # ── One fluence isosurface trace per variant ──────────────────────────────
    for j, (flu_log, fname) in enumerate(zip(prepped, fluence_names)):
        valid_log = flu_log[tissue_mask]

        iso_vals = np.linspace(
            np.percentile(flu_log[flu_log > flu_log.min()], 10),
            np.percentile(flu_log, 99), 5
        )
        traces.append(go.Isosurface(
            x=Xf, y=Yf, z=Zf,
            value=flu_log.flatten(),
            isomin=float(iso_vals[0]),
            isomax=float(iso_vals[-1]),
            surface_count=5,
            colorscale="Hot",
            showscale=(j == 0),  # only show colorbar once
            colorbar=dict(
                title=dict(text="log₁₀ Fluence Rate<br>(mW/cm²)", side="right"),
                thickness=15, len=0.6,
            ),
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=fname,
            visible=(j == 0),  # only combined visible initially
            opacity=0.25,     # Was 0.25
            lighting=dict(ambient=0.6, diffuse=0.6, specular=0.3, roughness=0.5),
        ))

    fig = go.Figure(data=traces)
    print(f"  Traces in figure after go.Figure: {len(fig.data)}")   # test line added
    fig = add_source_traces(fig, origin, mesh_center, arrow_length=20)
    fig._n_tissue_traces = n_tissue_traces_added  # attach for write_interactive_html
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
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: voxelize ─────────────────────────────────────────────────────
    print("=== Step 1: Building label volume ===")
    vol, origin, mesh_center = build_label_volume(TISSUES, VOXEL_RES, VOXEL_SIZE)

    # ── Step 2: compute source directions toward mesh center ──────────────────
    for cfg in SRC_WORLD_CONFIGS:
        d = np.array([0, 0, 0]) - np.array(cfg['world_pos'])
        cfg['srcdir'] = (d / np.linalg.norm(d)).tolist()

    # ── Step 3: place sources on tissue surface ───────────────────────────────
    PMCX_SOURCE_PLUS = find_surface_source_positions(
        vol, origin, VOXEL_SIZE, mesh_center, SRC_WORLD_CONFIGS
    )
    PMCX_SOURCE = [
        {'srcpos': s['srcpos'], 'srcdir': s['srcdir']}
        for s in PMCX_SOURCE_PLUS
    ]

    # Verify all sources are in tissue
    print("\nSource position verification:")
    for src in PMCX_SOURCE:
        sp = [int(round(x)) for x in src['srcpos']]
        sp_clipped = [np.clip(sp[i], 0, vol.shape[i]-1) for i in range(3)]
        label = vol[sp_clipped[0], sp_clipped[1], sp_clipped[2]]
        print(f"  Source at vox {sp}, label={label}, "
              f"dir={[f'{x:.3f}' for x in src['srcdir']]}")
        assert label > 0, f"Source still in background at {sp}!"

    # ── Step 4: add synovial fluid and wrapping layers ────────────────────────
    BONE_LABELS      = [t[1] for name, t in TISSUES.items() if "bone" in name]
    CARTILAGE_LABELS = [t[1] for name, t in TISSUES.items() if "cart" in name]
    MENISCUS_LABELS  = [t[1] for name, t in TISSUES.items() if "men"  in name]

    vol = add_synovial_fluid(
        vol,
        cartilage_labels=CARTILAGE_LABELS + MENISCUS_LABELS,
        bone_labels=BONE_LABELS,
        fluid_label=TISSUES["synovial"][1],
        dilation_vox=3
    )
    print(f"After synovial: {np.unique(vol)}")

    LAYER_CONFIGS_VOX = [
        (TISSUES["muscle"][1],  int(round(12 / VOXEL_SIZE))),
        (TISSUES["adipose"][1], int(round(6  / VOXEL_SIZE))),
        (TISSUES["skin"][1],    int(round(2  / VOXEL_SIZE))),
    ]
    vol = add_wrapping_layers(vol, LAYER_CONFIGS_VOX)
    print(f"After wrapping layers: {np.unique(vol)}")

    np.save("label_volume.npy", vol)
    print(f"Saved label_volume.npy shape={vol.shape} labels={np.unique(vol)}")

    # ── Step 5: fluence ───────────────────────────────────────────────────────
    print("\n=== Step 2: Fluence ===")

    if FLUENCE_OUTPUT is not None:
        fluence_combined = np.load("fluence_combined.npy")
        fluence_list = [np.load(f"fluence_src{i + 1}.npy")
                        for i in range(len(PMCX_SOURCE))]
    else:
        fluence_combined, fluence_list = run_pmcx(vol, TISSUES, PMCX_SOURCE)

    # Always built regardless of which branch was taken
    all_fluences = [fluence_combined] + fluence_list
    fluence_names = ["All Sources"] + [src['name'] for src in PMCX_SOURCE_PLUS]

    ''' CODE REMOVED TO CHECK CODE ABOVE
    if FLUENCE_OUTPUT is not None:
        print(f"  Loading saved fluence files")
        fluence_combined = np.load("fluence_combined.npy")
        fluence_list = [np.load(f"fluence_src{i + 1}.npy")
                        for i in range(len(PMCX_SOURCE))]
    else:
        fluence_combined, fluence_list = run_pmcx(vol, TISSUES, PMCX_SOURCE)

        # Build these BEFORE plot_results and write_interactive_html
        all_fluences = [fluence_combined] + fluence_list
        fluence_names = ["All Sources"] + [src['name'] for src in PMCX_SOURCE_PLUS]
    '''

    # ── Step 6: plot ──────────────────────────────────────────────────────────
    print("\n=== Step 3: Plotting ===")
    fig = plot_results(vol, fluence_combined, fluence_list, all_fluences, fluence_names,
                       TISSUES, origin, VOXEL_SIZE, mesh_center=mesh_center)

    output_path = "fluence_overlay.html"
    write_interactive_html(fig, TISSUES, output_path=output_path)
    print("  Done — fluence_overlay.html written.")

    # Open in default browser
    abs_path = os.path.abspath(output_path)
    webbrowser.open(f"file:///{abs_path}")

    end_time = time.perf_counter()
    print(f"Elapsed: {end_time - start_time:.2f} seconds")
