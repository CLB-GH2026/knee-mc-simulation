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
from scipy.ndimage import gaussian_filter, binary_dilation
from pathlib import Path
import webbrowser
import os

base_dir = Path()
mesh_dir = base_dir / 'Raw_Mesh_Files'

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

start_time = time.perf_counter()

def opt(mua, mus_prime, g, n):
    """Convert reduced scattering coefficient to transport scattering for pmcx."""
    return [mua, mus_prime / (1 - g), g, n]

# Map tissue name → (stl_path, label_id, optical_props [mua, mus, g, n])
# Optical properties at 808nm, units mm⁻¹, mus is transport scattering
TISSUES = {
    "synovial":     (None,                                   14, opt(0.0002, 0.01,  0.90, 1.36)),
    "skin":         (None,                                   13, opt(0.0046, 1.22,  0.79, 1.40)),
    "adipose":      (None,                                   12, opt(0.0057, 1.00,  0.90, 1.44)),
    "muscle":       (None,                                   11, opt(0.0180, 0.55,  0.93, 1.37)),
    "pat-cart":     (mesh_dir/"patella_lig_raw.stl",         10, opt(0.0050, 1.50,  0.90, 1.37)),
    "mtc-cart":     (mesh_dir/"tibia_cartilage_med_raw.stl",  9, opt(0.0050, 1.50,  0.90, 1.37)),
    "ltc-cart":     (mesh_dir/"tibia_cartilage_lat_raw.stl",  8, opt(0.0050, 1.50,  0.90, 1.37)),
    "fc-cart":      (mesh_dir/"femur_cartilage_raw.stl",      7, opt(0.0050, 1.50,  0.90, 1.37)),
    "mm-men":       (mesh_dir/"men_med_raw.stl",              6, opt(0.0060, 1.80,  0.90, 1.37)),
    "lm-men":       (mesh_dir/"men_lat_raw.stl",              5, opt(0.0060, 1.80,  0.90, 1.37)),
    "patella-bone": (mesh_dir/"patella_raw.stl",              4, opt(0.0130, 2.50,  0.92, 1.37)),
    "fibula-bone":  (mesh_dir/"fibula_raw.stl",               3, opt(0.0130, 2.50,  0.92, 1.37)),
    "tibia-bone":   (mesh_dir/"tibia_raw.stl",                2, opt(0.0130, 2.50,  0.92, 1.37)),
    "femur-bone":   (mesh_dir/"femur_raw.stl",                1, opt(0.0130, 2.50,  0.92, 1.37)),
}

# Source positions in centered world coordinates (mm), relative to mesh center (0,0,0)
SRC_WORLD_CONFIGS = [
    {'name': 'Source 1', 'world_pos': [-35,  60, 0], 'color': 'red'  },
    {'name': 'Source 2', 'world_pos': [ 35,  60, 0], 'color': 'green' },
    {'name': 'Source 3', 'world_pos': [  0, -60, 0], 'color': 'blue'},
]

VOXEL_SIZE = 1.0               # mm per voxel

# Physical grid dimensions in mm (from bounding box + padding)
GRID_DIMS_MM = (150, 140, 285)   # x, y, z in mm — edit these, not VOXEL_RES

# Compute VOXEL_RES automatically from physical size and voxel size
VOXEL_RES = tuple(int(round(d / VOXEL_SIZE)) for d in GRID_DIMS_MM)
print(f"VOXEL_RES: {VOXEL_RES}  (at VOXEL_SIZE={VOXEL_SIZE}mm)")

FLUENCE_OUTPUT = None          # None to run pmcx, or True to load saved files

# ─────────────────────────────────────────────────────────────────────────────
# 2. STL → VOXEL LABEL VOLUME
# ─────────────────────────────────────────────────────────────────────────────

def stl_to_voxels(mesh_path, label, origin, spacing, shape):
    """Ray-cast a closed STL mesh into a voxel volume."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if not mesh.is_watertight:
        print(f"  ⚠  {mesh_path} is not watertight — attempting repair")
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)

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
    """Merge all tissue STLs into one integer label volume."""
    all_verts = []
    for name, (path, label, _) in tissues.items():
        if path is not None:
            m = trimesh.load(path, force="mesh")
            all_verts.append(m.vertices)

    verts       = np.vstack(all_verts)
    mn          = verts.min(axis=0)
    mx          = verts.max(axis=0)
    mesh_center = (mn + mx) / 2.0
    grid_half   = np.array(res) * spacing / 2.0
    origin      = mesh_center - grid_half
    mesh_dims = mx - mn

    vol = np.zeros(res, dtype=np.uint8)
    for name, (path, label, _) in tissues.items():
        if path is not None:
            print(f"  Voxelizing {name} (label={label})...")
            layer = stl_to_voxels(path, label, origin, spacing, res)
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

    print(f"  Synovial fluid voxels: {fluid_mask.sum()}")
    result = vol.copy()
    result[fluid_mask] = fluid_label
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLACE SOURCES ON TISSUE SURFACE
# ─────────────────────────────────────────────────────────────────────────────

def find_surface_source_positions(vol, origin, spacing, mesh_center, src_configs):
    """Place sources just inside the tissue surface."""
    sources       = []
    tissue_coords = np.argwhere(vol > 0)

    for cfg in src_configs:
        intended_world = np.array(cfg['world_pos'])
        intended_vox   = (intended_world + mesh_center - origin) / spacing

        distances = np.linalg.norm(tissue_coords - intended_vox, axis=1)
        nearest   = tissue_coords[distances.argmin()]

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
# 7. PMCX SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_pmcx(vol, tissues, src_cfg,
             source_power_mw=50,
             wavelength_m=808e-9,
             modulation_hz=40,
             duty_cycle=0.75):
    """Run pmcx simulation and return fluence in mW/cm²."""

    h            = 6.626e-34
    c            = 3e8
    E_photon     = h * c / wavelength_m
    power_avg_W  = (source_power_mw * 1e-3) * duty_cycle
    Q_avg_per_s  = power_avg_W / E_photon

    # fluence (mm⁻²/ph) × Q (ph/s) × E_photon (J/ph) × 100 (mm²/cm²) × 1000 (mW/W)
    scale = Q_avg_per_s * E_photon * 100.0 * 1e3

    print(f"  Average power:  {power_avg_W*1e3:.2f} mW")

    max_label  = max(t[1] for t in tissues.values())
    prop_table = [[0, 0, 1, 1]] * (max_label + 1)
    for name, (path, label, opts) in tissues.items():
        prop_table[label] = opts

    cone_angle     = 32
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

    for i, src in enumerate(PMCX_SOURCE):
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
    11: "rgba(180,60,60,0.20)",     # muscle
    12: "rgba(255,220,150,0.15)",   # adipose
    13: "rgba(210,180,140,0.10)",   # skin
    14: "rgba(173,216,230,0.45)",   # synovial
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
                 plot_stride=4, mesh_center=None):
    """Build Plotly figure with tissue isosurfaces and fluence overlay."""

    s         = plot_stride
    vol_p     = vol[::s, ::s, ::s]
    spacing_p = spacing * s
    X, Y, Z   = make_coord_arrays(vol_p.shape, origin, spacing_p, center=mesh_center)
    Xf, Yf, Zf = X.flatten(), Y.flatten(), Z.flatten()

    def prep_fluence(flu, vol_p, name=""):
        flu_ds      = flu[::s, ::s, ::s]
        tissue_mask = vol_p > 0
        nonzero_flu = flu_ds[tissue_mask & (flu_ds > 0)]

        if len(nonzero_flu) == 0:
            print("  Warning: no valid fluence voxels found")
            return np.zeros_like(flu_ds)

        print(f"  [{name}] Nonzero tissue voxels: {len(nonzero_flu)} of {tissue_mask.sum()} "
              f"({100 * len(nonzero_flu) / tissue_mask.sum():.1f}%)")

        floor_val = np.percentile(nonzero_flu, 10)
        ceil_val = np.percentile(nonzero_flu, 99)

        flu_clamped = np.where(
            tissue_mask & (flu_ds >= floor_val),
            np.clip(flu_ds, floor_val, ceil_val),
            floor_val
        )
        flu_log = np.log10(flu_clamped)

        if smooth_sigma > 0:
            flu_log = gaussian_filter(flu_log, sigma=smooth_sigma)

        flu_log = np.where(tissue_mask, flu_log, np.log10(floor_val) - 1)
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
# 12. WRITE INTERACTIVE HTML
# ─────────────────────────────────────────────────────────────────────────────

def write_interactive_html(fig, tissues, output_path="fluence_overlay.html"):
    import json
    import re as _re

    sorted_tissues = sorted(tissues.items(), key=lambda kv: kv[1][1])
    n_tissues      = fig._n_tissue_traces
    n_fluence      = len(all_fluences)
    n_sources      = len(PMCX_SOURCE_PLUS) * 3
    n_total        = len(fig.data)

    assert n_tissues + n_fluence + n_sources == n_total, \
        f"Trace count mismatch: {n_tissues}+{n_fluence}+{n_sources}" \
        f"={n_tissues+n_fluence+n_sources} != {n_total}"

    def extract_opacity(label):
        """Extract opacity value from rgba string in TISSUE_COLORS."""
        color = TISSUE_COLORS.get(label, "rgba(200,200,200,1.0)")
        match = _re.search(r'rgba\([^,]+,[^,]+,[^,]+,([^)]+)\)', color)
        return float(match.group(1)) if match else 1.0

    tissue_info = [
        {
            "name": name,
            "trace_idx": i,
            "label": data[1],
            "default_visible": True,
            "default_opacity": extract_opacity(data[1])
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
    div_id_match = _re.search(r'<div id="([^"]+)"[^>]*class="plotly-graph-div"', html)
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
            max-width: 240px;
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
        .tissue-label {
            font-size: 11px;
            color: #e6edf3;
            width: 90px;
            flex-shrink: 0;
        }
        .opacity-slider {
            -webkit-appearance: none;
            width: 100%;
            height: 4px;
            border-radius: 2px;
            background: #30363d;
            outline: none;
            cursor: pointer;
        }
        .opacity-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #58a6ff;
            cursor: pointer;
        }
        .opacity-slider::-moz-range-thumb {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #58a6ff;
            cursor: pointer;
            border: none;
        }
        .opacity-value {
            font-size: 10px;
            color: #8b949e;
            width: 28px;
            text-align: right;
            flex-shrink: 0;
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
            <button onclick="setAll(1.0)">All On</button>
            <button onclick="setAll(0.0)">All Off</button>
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

        // Track opacity per trace (0.0 to 1.0)
        let opacityState = new Array(nTotal).fill(1.0);

        // Fluence traces — only first visible initially
        let visState = new Array(nTotal).fill(true);
        for (let j = 0; j < nFluence; j++) {
            visState[nTissues + j] = (j === 0);
        }

        // Apply initial visibility after plot renders
        plotDiv.on('plotly_afterplot', function() {
            plotDiv.removeAllListeners('plotly_afterplot');
            Plotly.update(plotDiv, { visible: visState.map(v => !!v) });
        });

        // Build tissue opacity sliders
        const container = document.getElementById('tissue-toggles');
        tissueInfo.forEach((t, i) => {
            const row = document.createElement('div');
            row.className = 'tissue-row';

            const lbl = document.createElement('span');
            lbl.className = 'tissue-label';
            lbl.textContent = t.name;

            const slider = document.createElement('input');
            slider.type  = 'range';
            slider.min   = '0';
            slider.max   = '1';
            slider.step  = '0.05';
            slider.value = t.default_opacity;
            slider.className = 'opacity-slider';
            slider.id    = 'slider-' + i;

            const valDisplay = document.createElement('span');
            valDisplay.className = 'opacity-value';
            valDisplay.id = 'val-' + i;
            valDisplay.textContent = parseFloat(t.default_opacity).toFixed(2);

            slider.oninput = () => {
                const opacity = parseFloat(slider.value);
                valDisplay.textContent = opacity.toFixed(2);
                opacityState[i] = opacity;
                applyOpacity(i, opacity);
            };

            row.appendChild(lbl);
            row.appendChild(slider);
            row.appendChild(valDisplay);
            container.appendChild(row);

            // Set initial opacity state
            opacityState[i] = parseFloat(t.default_opacity);
        });

        // Build fluence dropdown
        const sel = document.getElementById('fluence-dropdown');
        fluenceNames.forEach((name, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = name;
            sel.appendChild(opt);
        });

        function applyOpacity(traceIdx, opacity) {
            // Hide trace if opacity is 0, show if > 0
            visState[traceIdx] = opacity > 0;
            Plotly.update(plotDiv,
                { opacity: opacity },
                [traceIdx]
            );
            Plotly.update(plotDiv,
                { visible: visState.map(v => !!v) }
            );
        }

        function setAll(opacity) {
            tissueInfo.forEach((t, i) => {
                opacityState[i] = opacity;
                const slider = document.getElementById('slider-' + i);
                const valDisplay = document.getElementById('val-' + i);
                slider.value = opacity;
                valDisplay.textContent = opacity.toFixed(2);
                visState[i] = opacity > 0;
                Plotly.update(plotDiv, { opacity: opacity }, [i]);
            });
            Plotly.update(plotDiv, { visible: visState.map(v => !!v) });
        }

        function switchFluence(idx) {
            const active = parseInt(idx);
            for (let j = 0; j < nFluence; j++) {
                visState[nTissues + j] = (j === active);
            }
            Plotly.update(plotDiv, { visible: visState.map(v => !!v) });
        }
    </script>""")

    html = html.replace('<body>', '<body>\n' + controls_html, 1)
    html = html.replace('</body>', controls_js + '\n</body>', 1)

    with open(output_path, 'w') as f:
        f.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# 13. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: Build label volume from STL files ─────────────────────────────
    print("=== Step 1: Building label volume ===")
    vol, origin, mesh_center = build_label_volume(TISSUES, VOXEL_RES, VOXEL_SIZE)

    # ── Step 2: Compute source directions toward mesh center ──────────────────
    for cfg in SRC_WORLD_CONFIGS:
        d = np.array([0, 0, 0]) - np.array(cfg['world_pos'])
        cfg['srcdir'] = (d / np.linalg.norm(d)).tolist()

    # ── Step 3: Place sources on tissue surface ───────────────────────────────
    PMCX_SOURCE_PLUS = find_surface_source_positions(
        vol, origin, VOXEL_SIZE, mesh_center, SRC_WORLD_CONFIGS
    )
    PMCX_SOURCE = [{'srcpos': s['srcpos'], 'srcdir': s['srcdir']}
                   for s in PMCX_SOURCE_PLUS]

    # Verify all sources are inside tissue
    print("\nSource position verification:")
    for src in PMCX_SOURCE:
        sp         = [int(round(x)) for x in src['srcpos']]
        sp_clipped = [np.clip(sp[i], 0, vol.shape[i]-1) for i in range(3)]
        label      = vol[sp_clipped[0], sp_clipped[1], sp_clipped[2]]
        print(f"  vox={sp}, label={label}, dir={[f'{x:.3f}' for x in src['srcdir']]}")
        assert label > 0, f"Source still in background at {sp}!"

    # ── Step 4: Add synovial fluid then wrapping layers ───────────────────────
    BONE_LABELS      = [t[1] for name, t in TISSUES.items() if "bone"  in name]
    CARTILAGE_LABELS = [t[1] for name, t in TISSUES.items() if "cart"  in name]
    MENISCUS_LABELS  = [t[1] for name, t in TISSUES.items() if "men"   in name]

    vol = add_synovial_fluid(
        vol,
        cartilage_labels=CARTILAGE_LABELS + MENISCUS_LABELS,
        bone_labels=BONE_LABELS,
        fluid_label=TISSUES["synovial"][1],
        dilation_vox=3
    )

    LAYER_CONFIGS_VOX = [
        (TISSUES["muscle"][1],  int(round(12 / VOXEL_SIZE))),
        (TISSUES["adipose"][1], int(round(6  / VOXEL_SIZE))),
        (TISSUES["skin"][1],    int(round(2  / VOXEL_SIZE))),
    ]
    vol = add_wrapping_layers(vol, LAYER_CONFIGS_VOX)

    np.save("label_volume.npy", vol)
    print(f"Label volume: shape={vol.shape}, labels={np.unique(vol)}")

    # ── Step 5: Run or load fluence ───────────────────────────────────────────
    print("\n=== Step 2: Fluence ===")
    if FLUENCE_OUTPUT is not None:
        fluence_combined = np.load("fluence_combined.npy")
        fluence_list     = [np.load(f"fluence_src{i+1}.npy")
                            for i in range(len(PMCX_SOURCE))]
    else:
        fluence_combined, fluence_list = run_pmcx(vol, TISSUES, PMCX_SOURCE)

    all_fluences  = [fluence_combined] + fluence_list
    fluence_names = ["All Sources"] + [src['name'] for src in PMCX_SOURCE_PLUS]

    # ── Step 6: Plot and write HTML ───────────────────────────────────────────
    print("\n=== Step 3: Plotting ===")
    fig = plot_results(vol, fluence_combined, fluence_list, all_fluences, fluence_names,
                       TISSUES, origin, VOXEL_SIZE, mesh_center=mesh_center)

    output_path = "fluence_overlay.html"
    write_interactive_html(fig, TISSUES, output_path=output_path)
    print("Done — fluence_overlay.html written.")

    webbrowser.open(f"file:///{os.path.abspath(output_path)}")

    end_time = time.perf_counter()
    print(f"Elapsed: {end_time - start_time:.2f} seconds")