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
from scipy.ndimage import gaussian_filter
from pathlib import Path

base_dir = Path()
mesh_dir = base_dir / 'Mesh Files'

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────

start_time = time.perf_counter()

# Map tissue name → (stl_path, label_id, optical_props [mua, mus, g, n])
# Optical properties at 750 nm (example values — replace with your own)
TISSUES = {
    "mcl-lig":     ("mcl.stl",                  11, [0.07, 12.5, 0.9, 1.37]),
    "acl-lig":     ("acl.stl",                  10, [0.07, 12.5, 0.9, 1.37]),
    "lcl-lig":     ("lcl.stl",                  9, [0.07, 12.5, 0.9, 1.37]),
    "pcl-lig":     ("pcl.stl",                  8, [0.07, 12.5, 0.9, 1.37]),
    "mtc-cart":    ("med_tibial_cartilage.stl", 7, [0.07, 12.5, 0.9, 1.37]),
    "ltc-cart":    ("lat_tibial_cartilage.stl", 6, [0.07, 12.5, 0.9, 1.37]),
    "fc-cart":     ("femur_cartilage.stl",      5, [0.07, 12.5, 0.9, 1.37]),
    "mm-men":      ("med_meniscus.stl",         4, [0.07, 12.5, 0.9, 1.37]),
    "lm-men":      ("lat_meniscus.stl",         3, [0.07, 12.5, 0.9, 1.37]),
    "tibia-bone":  ("tibia.stl",                2, [0.14, 15.8, 0.9, 1.37]),
    "femur-bone":  ("femur_NoHoles.stl",        1, [0.14, 15.8, 0.9, 1.37]),   #femur.stl (has holes to show inside lig attachment??
}

SRC_FRAC =[[1.0, 0.7, 0.23], [1.0, 0.7, 0.60], [0.0, 0.7, 0.4]]

VOXEL_RES  = (100, 100, 100)   # (nx, ny, nz) — increase for finer detail
VOXEL_SIZE = 1.0               # mm per voxel

# pmcx source — pencil beam pointing in +z, centered on top face
# Adjust srcpos/srcdir to match your geometry
PMCX_SOURCE = [
    {'srcpos': [SRC_FRAC[0][i] * VOXEL_RES[i] for i in range(3)],'srcdir': [-1,0,0]},
    {'srcpos': [SRC_FRAC[1][i] * VOXEL_RES[i] for i in range(3)],'srcdir': [-1,0,0]},
    {'srcpos': [SRC_FRAC[2][i] * VOXEL_RES[i] for i in range(3)],'srcdir': [1,0,0]},
]
PMCX_SOURCE_PLUS = [
    {'srcpos': [SRC_FRAC[0][i] * VOXEL_RES[i] for i in range(3)],'srcdir': [-1,0,0], 'color': 'red', 'name': 'Source 1'},
    {'srcpos': [SRC_FRAC[1][i] * VOXEL_RES[i] for i in range(3)],'srcdir': [-1,0,0], 'color': 'blue', 'name': 'Source 2'},
    {'srcpos': [SRC_FRAC[2][i] * VOXEL_RES[i] for i in range(3)],'srcdir': [1,0,0], 'color': 'green', 'name': 'Source 3'},
]
arrow_length = 5
n_rays = 8

for src in PMCX_SOURCE:
    for i, v in enumerate(src['srcpos']):
        assert 0 <= v <= VOXEL_RES[i], f"srcpos[{i}]={v} out of bounds {VOXEL_RES[i]}"

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


def build_label_volume(tissues: dict, res: tuple, spacing: float) -> np.ndarray:
    """
    Merge all tissue STLs into one integer label volume.
    Later tissues (higher label) overwrite earlier ones where they overlap.
    """
    # Determine bounding box across all meshes
    all_verts = []
    meshes_loaded = {}
    for name, (path, label, _) in tissues.items():
        m = trimesh.load(path, force="mesh")
        meshes_loaded[name] = m
        all_verts.append(m.vertices)

    verts = np.vstack(all_verts)
    mn = verts.min(axis=0) - spacing

    # override res to honour requested grid
    shape = res

    print(f"Global bounding box: {mn}  →  {mn + np.array(shape)*spacing}")

    vol = np.zeros(shape, dtype=np.uint8)
    for name, (path, label, _) in tissues.items():
        print(f"  Voxelizing {name} (label={label})…")
        layer = stl_to_voxels(path, label, mn, spacing, shape)
        mask  = layer > 0
        vol[mask] = layer[mask]

    return vol, mn


# ─────────────────────────────────────────────────────────────────────────────
# 3. PMCX SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_pmcx(vol: np.ndarray, tissues: dict, src_cfg: dict) -> np.ndarray:
    """
    Build pmcx config from label volume + tissue optical props, run, return fluence.
    """
    # prop row 0 = background (air/void)
    prop = [[0, 0, 1, 1]]
    for name in sorted(tissues, key=lambda k: tissues[k][1]):
        prop.append(tissues[name][2])

    # FWHM angle to half-angle in radians
    cone_angle = 32
    global half_angle_rad
    half_angle_rad = np.deg2rad(cone_angle/2)

    cfg = {
        "nphoton": 1e6,
        "srctype": 'cone',
        'srcparam1': [half_angle_rad, 0, 0, 0],
        "vol":    vol.astype(np.uint8),
        "prop":   prop,
        "tstart": 0,
        "tend":   5e-9,
        "tstep":  5e-9,
        "unitinmm": VOXEL_SIZE,
        "autopilot": 1,
        "gpuid":  1,        # set to 0 for CPU fallback
        "issavedet": 0,
    }
    cfg.update(src_cfg)

    print("Running pmcx…")

    combined_flux = None
    for src in PMCX_SOURCE:
        cfg['srcpos'] = src['srcpos']
        cfg['srcdir'] = src['srcdir']
        res = pmcx.run(cfg)

        if combined_flux is None:
            combined_flux = res['flux'].copy()
        else:
            combined_flux += res['flux']

    # fluence shape: (nx, ny, nz, ntime) — squeeze time
    fluence = combined_flux.squeeze()
    print(f"  Fluence shape: {fluence.shape},  max={fluence.max():.3e}")
    return fluence


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOTLY VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

# Distinct colours for up to 11 tissue types
TISSUE_COLORS = [
    "rgba(128,128,128,1.00)",   # femur bone - dark gray
    "rgba(128,128,128,1.00)",   # tibia bone - dark gray
    "rgba(255,153,51,1.00)",    # lat meniscus - orange
    "rgba(255,153,51,1.00)",    # med meniscus - orange
    "rgba(0,0,255,1.00)",       # femur cartilage - blue
    "rgba(0,0,255,1.00)",       # lat cartilage - blue
    "rgba(0,0,255,1.00)",       # med cartilage - blue
    "rgba(255,0,0,1.00)",       # pcl — red
    "rgba(255,0,0,1.00)",       # lcl — red
    "rgba(255,0,0,1.00)",       # acl — red
    "rgba(255,0,0,1.00)",       # mcl — red
]

def make_coord_arrays(shape, origin, spacing):
    nx, ny, nz = shape
    x = origin[0] + (np.arange(nx) + 0.5) * spacing
    y = origin[1] + (np.arange(ny) + 0.5) * spacing
    z = origin[2] + (np.arange(nz) + 0.5) * spacing
    return np.meshgrid(x, y, z, indexing="ij")


def plot_results(vol: np.ndarray, fluence: np.ndarray,
                 tissues: dict, origin: np.ndarray, spacing: float,
                 smooth_sigma: float = 1.0):
    """
    Plotly figure with:
      • One semi-transparent Isosurface per tissue label
      • Fluence Isosurface coloured by log10(fluence), hot colorscale
    """
    X, Y, Z = make_coord_arrays(vol.shape, origin, spacing)

    # log-fluence (clamp to avoid -inf)
    flu_log = np.log10(np.maximum(fluence, fluence[fluence > 0].min() * 1e-3))
    if smooth_sigma > 0:
        flu_log = gaussian_filter(flu_log, sigma=smooth_sigma)

    traces = []

    # ── Tissue isosurfaces ────────────────────────────────────────────────────
    sorted_tissues = sorted(tissues.items(), key=lambda kv: kv[1][1])
    for i, (name, (path, label, _)) in enumerate(sorted_tissues):
        tissue_mask = (vol == label).astype(float)
        tissue_mask = gaussian_filter(tissue_mask, sigma=0.8)
        color = TISSUE_COLORS[i % len(TISSUE_COLORS)]

        traces.append(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=tissue_mask.flatten(),
            isomin=0.4, isomax=0.6,
            surface_count=1,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name=name,
            opacity=0.85,   # last 0.85
            lighting=dict(ambient=0.7, diffuse=0.5, specular=0.2),
        ))

    # ── Fluence isosurface ────────────────────────────────────────────────────
    fmin, fmax = flu_log.min(), flu_log.max()
    # show 5 iso-levels spanning the dynamic range
    n_iso = 5
#    iso_vals = np.linspace(fmin + (fmax - fmin) * 0.15, fmax * 0.95, n_iso)
    iso_vals = np.linspace(np.percentile(flu_log[flu_log > flu_log.min()], 10),
                           np.percentile(flu_log, 99), n_iso)

    traces.append(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=flu_log.flatten(),
        isomin=float(iso_vals[0]),
        isomax=float(iso_vals[-1]),
        surface_count=n_iso,
        colorscale="Hot",
        reversescale=False,
        showscale=True,
        colorbar=dict(
            title=dict(text="log₁₀ Fluence<br>(mm⁻²)", side="right"),
            thickness=15, len=0.6,
        ),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="Fluence",
        opacity=0.0,   # Was 0.25
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.3, roughness=0.5),
    ))

    fig = go.Figure(data=traces)
    '''
    # --- Loop over sources ---
    for src in PMCX_SOURCE_PLUS:
        srcpos = np.array(src['srcpos'])
        srcdir = np.array(src['srcdir'])
        color = src['color']
        name = src['name']
        arrow_end = srcpos + srcdir * arrow_length

        # Source position marker
        fig.add_trace(go.Scatter3d(
            x=[srcpos[0]], y=[srcpos[1]], z=[srcpos[2]],
            mode='markers',
            marker=dict(size=8, color=color, symbol='diamond'),
            name=name
        ))
        Let's only pass the source marker info for now to see if it works
        # Source direction line
        fig.add_trace(go.Scatter3d(
            x=[srcpos[0], arrow_end[0]],
            y=[srcpos[1], arrow_end[1]],
            z=[srcpos[2], arrow_end[2]],
            mode='lines',
            line=dict(color=color, width=4),
            name=f'{name} direction',
            showlegend=True
        ))

        # Source direction tip
        fig.add_trace(go.Scatter3d(
            x=[arrow_end[0]],
            y=[arrow_end[1]],
            z=[arrow_end[2]],
            mode='markers',
            marker=dict(size=6, color=color, symbol='circle'),
            showlegend=False
        ))

        # Cone edge rays
        for i in range(n_rays):
            angle = 2 * np.pi * i / n_rays

            # Build a perpendicular vector to srcdir
            ref = np.array([1, 0, 0]) if abs(srcdir[0]) < 0.9 else np.array([0, 1, 0])
            perp_x = np.cross(srcdir, ref)
            perp_x /= np.linalg.norm(perp_x)
            perp_y = np.cross(srcdir, perp_x)

            perp = np.cos(angle) * perp_x + np.sin(angle) * perp_y
            ray_dir = np.sin(half_angle_rad) * perp + np.cos(half_angle_rad) * srcdir
            ray_dir /= np.linalg.norm(ray_dir)
            ray_end = srcpos + ray_dir * arrow_length

            fig.add_trace(go.Scatter3d(
                x=[srcpos[0], ray_end[0]],
                y=[srcpos[1], ray_end[1]],
                z=[srcpos[2], ray_end[2]],
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                showlegend=(i == 0),
                name=f'{name} cone' if i == 0 else ''
            ))
        '''
    fig.update_layout(
        title="Multi-tissue voxel volume + pmcx fluence",
        scene=dict(
            xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
            bgcolor="#0d1117",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            camera=dict(
                eye=dict(x=0.5, y=0.0, z=2.0),       # Camera position
                center=dict(x=0, y=0, z=0),     # Point camera looks at
                up=dict(x=0, y=-1, z=0)          # "Up" Direction
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
    vol, origin = build_label_volume(TISSUES, VOXEL_RES, VOXEL_SIZE)
    np.save("label_volume.npy", vol)

    print(f"  Saved label_volume.npy  shape={vol.shape}  labels={np.unique(vol)}")

    # ── Step 2: fluence ──────────────────────────────────────────────────────
    print("\n=== Step 2: Fluence ===")

    if FLUENCE_OUTPUT is not None:
        print(f"  Loading saved fluence from {FLUENCE_OUTPUT}")
        fluence = np.load(FLUENCE_OUTPUT)
    else:
        fluence = run_pmcx(vol, TISSUES, PMCX_SOURCE)
        np.save("fluence.npy", fluence)
        print("  Saved fluence.npy")

    # ── Step 3: plot ─────────────────────────────────────────────────────────
    print("\n=== Step 3: Plotting ===")
    fig = plot_results(vol, fluence, TISSUES, origin, VOXEL_SIZE)
    fig.write_html("fluence_overlay.html")   # save interactive HTML
    fig.show()
    #config={
    #    'displayModeBar': True,
    #    'responsive': True,
    #    'plotlyServerURL': None
    #})                               # open in browser / Jupyter


    print("  Done — fluence_overlay.html written.")
    end_time = time.perf_counter()
    print(f"Elapsed:  {end_time - start_time:.2f} seconds")