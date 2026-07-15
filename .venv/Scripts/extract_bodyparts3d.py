"""
extract_bodyparts3d.py
======================
Extracts bones from the BodyParts3D dataset and renames them to match
the file naming convention expected by the PBM Monte Carlo simulation
pipelines in:
    shoulder-mc-simulation   (SHO###)
    elbow-mc-simulation      (ELB###)
    lower_back-mc-simulation (LBK###)

TWO SOURCE STRATEGIES
---------------------
A) GitHub STL mirror  (Kevin-Mattheus-Moerman/BodyParts3D)
   - 180 pre-converted STL files; subset of the full dataset.
   - Use these when available — no conversion needed.
   - Assumed to be already cloned to a local path you supply.

B) DBCLS OBJ zip  (dbarchive.biosciencedbc.jp)
   - Full dataset (1500+ structures) as OBJ files in a 136 MB zip.
   - Downloaded automatically if any needed files are missing from the
     GitHub mirror.
   - OBJ files are converted to STL by this script using trimesh.

USAGE
-----
# Use the GitHub clone you already have (will auto-download extras):
python extract_bodyparts3d.py --bp3d_dir path/to/BodyParts3D --joint all

# Single joint, specific subject directory:
python extract_bodyparts3d.py --bp3d_dir ./BodyParts3D --joint shoulder --out_dir ./Raw_Mesh_Files_SHO001

# Specify left or right side (default: right):
python extract_bodyparts3d.py --bp3d_dir ./BodyParts3D --joint elbow --side left

DEPENDENCIES
------------
pip install trimesh numpy scipy scikit-image
  (scikit-image + scipy.ndimage drive the voxel-based bone cropping and the
   cartilage/labrum synthesis that fills the joints' soft-tissue targets.)

NOTES
-----
- All geometry is for a single adult male (the BodyParts3D reference model).
  For a multi-subject study, use TotalSegmentator on individual CT scans.
- The script picks the RIGHT side by default to match the coordinate
  convention used in all three simulation pipelines (+X = lateral right).
- Sacrum: only one instance in BodyParts3D (not left/right).
- Intervertebral discs: the DBCLS zip uses a generic unilateral model.
- The GitHub mirror has lumbar vertebrae L1-L5 and both scapulae/clavicles
  but is MISSING: humerus, radius, ulna, sacrum, and all lumbar discs.
  Those are auto-fetched from DBCLS.

LICENSE
-------
BodyParts3D, Copyright (c) 2008 Database Center for Life Science (DBCLS)
Licensed under CC Attribution-Share Alike 2.1 Japan.
Cite: Mitsuhashi N et al., Nucleic Acids Res. 2009 Jan;37(Database):D782-5.
"""

import argparse
import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage

# ─────────────────────────────────────────────────────────────────────────────
# MASTER MAPPING
# fma_id -> (bp_code, plain_name, github_stl_available)
#
# bp_code:  filename inside the DBCLS OBJ zip  →  BP{code}.obj
# github:   True if the FMA STL is in the GitHub mirror stl/ subdirectory
# ─────────────────────────────────────────────────────────────────────────────
STRUCTURES = {
    # ── Shoulder ─────────────────────────────────────────────────────────────
    "right_humerus":   ("FMA23130", "BP9206", False),
    "left_humerus":    ("FMA23131", "BP9191", False),
    "right_scapula":   ("FMA13395", "BP9101", True),
    "left_scapula":    ("FMA13396", "BP9121", True),
    "right_clavicle":  ("FMA13322", "BP9271", True),
    "left_clavicle":   ("FMA13323", "BP8841", True),

    # ── Elbow (radius + ulna; shares humerus with shoulder) ──────────────────
    "right_radius":    ("FMA23464", "BP8464", False),
    "left_radius":     ("FMA23465", "BP9059", False),
    "right_ulna":      ("FMA23467", "BP8233", False),
    "left_ulna":       ("FMA23468", "BP9070", False),

    # ── Lower back ────────────────────────────────────────────────────────────
    "L1_vertebra":     ("FMA13072", "BP8948", True),
    "L2_vertebra":     ("FMA13073", "BP8995", True),
    "L3_vertebra":     ("FMA13074", "BP8227", True),
    "L4_vertebra":     ("FMA13075", "BP8133", True),
    "L5_vertebra":     ("FMA13076", "BP8280", True),
    "sacrum":          ("FMA16202", "BP9174", False),
    "L1L2_disc":       ("FMA16033", "BP8891", False),
    "L2L3_disc":       ("FMA16034", "BP8683", False),
    "L3L4_disc":       ("FMA16035", "BP8046", False),
    "L4L5_disc":       ("FMA16036", "BP8261", False),
    "L5S1_disc":       ("FMA16037", "BP8781", False),
}

# ─────────────────────────────────────────────────────────────────────────────
# PER-JOINT FILE LISTS
# Maps: pipeline STL filename -> structure key in STRUCTURES above
# ─────────────────────────────────────────────────────────────────────────────
JOINT_FILES = {
    "shoulder": {
        "humerus_raw.stl":           "humerus",        # resolved to right_/left_ by --side
        "scapula_raw.stl":           "scapula",
        "clavicle_raw.stl":          "clavicle",
        "humeral_cartilage_raw.stl": None,              # not in BodyParts3D — placeholder
        "glenoid_cartilage_raw.stl": None,
        "labrum_raw.stl":            None,
    },
    "elbow": {
        "humerus_distal_raw.stl":          "humerus",   # full humerus; crop in pipeline
        "radius_raw.stl":                  "radius",
        "ulna_raw.stl":                    "ulna",
        "capitellum_cartilage_raw.stl":    None,
        "radial_head_cartilage_raw.stl":   None,
        "trochlear_cartilage_raw.stl":     None,
        "annular_lig_raw.stl":             None,
    },
    "lower_back": {
        "L3_raw.stl":      "L3_vertebra",
        "L4_raw.stl":      "L4_vertebra",
        "L5_raw.stl":      "L5_vertebra",
        "S1_raw.stl":      "sacrum",
        "L3L4_disc_raw.stl": "L3L4_disc",
        "L4L5_disc_raw.stl": "L4L5_disc",
        "L5S1_disc_raw.stl": "L5S1_disc",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# JOINT-LOCAL RECENTRING
# BodyParts3D geometry is expressed in a whole-body coordinate frame (the joint
# sits ~1.2 m from the origin). The MC pipeline, however, expects each subject's
# meshes in a JOINT-LOCAL frame with the articulation at the origin — the same
# convention the knee (OKS) meshes follow. Without this the meshes fall outside
# the simulation grid and nothing rasterises.
#
# For each joint we name the two primary articulating bones; the articulation
# centre is the midpoint of their closest surface points. All bone meshes are
# translated so that point lands on the origin; the soft-tissue placeholder
# cubes are written at the origin already, so they need no translation.
# ─────────────────────────────────────────────────────────────────────────────
JOINT_ARTICULATION = {
    "shoulder": ("humerus_raw.stl", "scapula_raw.stl"),      # glenohumeral
    "elbow":    ("humerus_distal_raw.stl", "ulna_raw.stl"),  # humero-ulnar hinge
    # lower_back has no single articulation pair — see JOINT_RECENTER instead.
}

# For joints that aren't a two-bone articulation (the lumbar column), recentre
# on the centroid of these target structures instead of the closest-points
# midpoint. Centring on the discs keeps the PBM targets in the middle of the
# grid; recentring on the whole column would be dragged inferior by the large
# sacrum, pushing the discs high in the grid.
JOINT_RECENTER = {
    "lower_back": ["L3L4_disc_raw.stl", "L4L5_disc_raw.stl", "L5S1_disc_raw.stl"],
}

# After recentring, bones are cropped to a sphere of this radius about the joint
# origin. BodyParts3D provides full-length bones (~300 mm); for a compact joint,
# the long shaft would otherwise bias the bounding-box midpoint that the pipeline
# uses to centre the grid, pushing the joint to the grid edge. Cropping keeps the
# joint region of every bone and centres the grid on it — the same effect the
# knee's pre-cropped OKS meshes have.
#   • shoulder/elbow: 90 mm sphere comfortably covers their grids.
#   • lower_back: 100 mm about the disc centroid — keeps the full L3–S1 span
#     (~150 mm) while capping the lower sacrum, which extends ~145 mm below the
#     discs (outside the PBM zone) and would otherwise drag the bounding-box
#     midpoint — and thus the grid centre — well inferior to the disc targets.
JOINT_CROP_RADIUS_MM = {
    "shoulder":   90.0,
    "elbow":      90.0,
    "lower_back": 100.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# SOFT-TISSUE SYNTHESIS
# BodyParts3D has no articular cartilage, labrum, or joint ligaments (it only
# segments costal/nasal/thyroid cartilage). Rather than leave placeholder cubes,
# we synthesise these targets from the bone geometry we do have:
#   • "shell": a thin cartilage layer, offset outward from the joint-facing bone
#     surface (voxel dilation minus bone), optionally split lateral/medial.
#   • "ring": a fibrocartilage torus (glenoid labrum, annular ligament).
# These are modelling approximations for MC fluence *targets*, not measured
# geometry — replace with segmented MRI if/when available. Radii/thicknesses are
# in mm; x_side "lat"/"med" splits a distal-humerus shell (right side: +X = lat).
# ─────────────────────────────────────────────────────────────────────────────
SOFT_TISSUE_SYNTHESIS = {
    "shoulder": {
        "humeral_cartilage_raw.stl": {"kind": "shell", "bone": "humerus_raw.stl",
                                       "thickness": 1.5, "radius": 26},
        "glenoid_cartilage_raw.stl": {"kind": "shell", "bone": "scapula_raw.stl",
                                       "thickness": 1.5, "radius": 15},
        "labrum_raw.stl":            {"kind": "ring", "bone": "scapula_raw.stl",
                                       "major": 13.0, "minor": 2.5, "center": "origin",
                                       "axis": "toward_head", "head_bone": "humerus_raw.stl",
                                       "head_radius": 28},
    },
    "elbow": {
        "capitellum_cartilage_raw.stl":  {"kind": "shell", "bone": "humerus_distal_raw.stl",
                                          "thickness": 1.3, "radius": 16, "x_side": "lat"},
        "trochlear_cartilage_raw.stl":   {"kind": "shell", "bone": "humerus_distal_raw.stl",
                                          "thickness": 1.3, "radius": 16, "x_side": "med"},
        "radial_head_cartilage_raw.stl": {"kind": "shell", "bone": "radius_raw.stl",
                                          "thickness": 1.2, "radius": 12, "center": "head",
                                          "head_radius": 15},
        "annular_lig_raw.stl":           {"kind": "ring", "bone": "radius_raw.stl",
                                          "major": 11.0, "minor": 2.0, "center": "head",
                                          "head_radius": 15, "axis": "long"},
    },
}

DBCLS_ZIP_URL = (
    "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/isa_BP3D_4.0_obj_99.zip"
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def resolve_key(base_key, side):
    """Turn e.g. 'humerus' + 'right' into 'right_humerus'."""
    return f"{side}_{base_key}"


def load_mesh(path):
    """Load any mesh file trimesh supports and return a single Trimesh."""
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    return mesh


def repair_and_save(mesh, out_path):
    """Attempt watertight repair and save as STL."""
    if not mesh.is_watertight:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)

    status = "watertight" if mesh.is_watertight else "NOT watertight (repair attempted)"
    mesh.export(str(out_path))
    print(f"    Saved: {out_path.name}  ({len(mesh.faces):,} faces)  [{status}]")
    return mesh.is_watertight


def obj_to_stl(obj_path, out_path):
    """Convert an OBJ file to STL."""
    mesh = load_mesh(obj_path)
    return repair_and_save(mesh, out_path)


def _articulation_center(mesh_a, mesh_b):
    """
    Midpoint of the closest pair of surface points between two meshes — a
    robust, landmark-free proxy for a joint's articulation centre. Queries the
    lower-vertex-count mesh against the other's surface for speed.
    """
    if len(mesh_a.vertices) > len(mesh_b.vertices):
        mesh_a, mesh_b = mesh_b, mesh_a
    closest, dist, _ = trimesh.proximity.closest_point(mesh_b, mesh_a.vertices)
    i = int(np.argmin(dist))
    return (mesh_a.vertices[i] + closest[i]) / 2.0


# BodyParts3D → pipeline axis transform.
# BodyParts3D (assets/.../coordinate_system.png): +X = left, +Y = posterior,
# +Z = superior. The MC pipeline (knee/OKS convention) expects +X = lateral,
# +Y = anterior, +Z = superior. Reconciling the two while preserving handedness
# — a proper 180° rotation about Z — negates X and Y and leaves Z. This maps
# anterior → +Y, superior → +Z, and the subject's RIGHT → +X, so for a
# right-side joint +X is lateral, matching the hardcoded source configs in the
# simulation scripts. Left-side extractions keep this same proper rotation
# (geometry is not mirrored), so a left-side run would need the source-config
# X-signs negated — revisit *_default_src_configs before running left joints.
_BP3D_TO_PIPELINE = np.array([
    [-1.0,  0.0, 0.0, 0.0],
    [ 0.0, -1.0, 0.0, 0.0],
    [ 0.0,  0.0, 1.0, 0.0],
    [ 0.0,  0.0, 0.0, 1.0],
])


def orient_and_recenter_joint(joint, out_dir, file_map):
    """
    Convert the joint's bone meshes from BodyParts3D whole-body coordinates into
    the pipeline's joint-local frame: rotate to pipeline axes (−X, −Y, +Z), then
    translate so the articulation sits at the origin (matching the knee/OKS
    convention). Placeholder soft-tissue cubes are already at the origin and are
    symmetric about it, so the rotation and translation leave them in place.

    Idempotent: each run re-extracts the bones in world coordinates before this
    pass, so re-running recomputes the same transform from scratch.
    """
    out_dir = Path(out_dir)
    bone_files = [n for n, key in file_map.items() if key is not None]
    if not bone_files:
        return None

    # Load every bone and rotate it into the pipeline axis frame.
    meshes = {}
    for n in bone_files:
        m = load_mesh(out_dir / n)
        m.apply_transform(_BP3D_TO_PIPELINE)
        meshes[n] = m

    # Recentring origin, computed in the already-rotated frame.
    pair = JOINT_ARTICULATION.get(joint)
    recenter_files = JOINT_RECENTER.get(joint)
    if recenter_files and all(f in meshes for f in recenter_files):
        verts = np.vstack([meshes[f].vertices for f in recenter_files])
        center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
        basis = "centroid of target structures (" + ", ".join(recenter_files) + ")"
    elif pair and all(p in meshes for p in pair):
        center = _articulation_center(meshes[pair[0]], meshes[pair[1]])
        basis = f"articulation of {pair[0]} ↔ {pair[1]}"
    else:
        # Fallback: centroid of the combined bone bounding box.
        verts = np.vstack([m.vertices for m in meshes.values()])
        center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
        basis = "combined bone bounding-box centre"

    crop_r = JOINT_CROP_RADIUS_MM.get(joint) if isinstance(JOINT_CROP_RADIUS_MM, dict) \
        else JOINT_CROP_RADIUS_MM
    print(f"    Reframe to pipeline axes (−X, −Y, +Z) + recentre ({basis})")
    print(f"      articulation origin shifted by {(-center).round(1)} mm")
    if crop_r:
        print(f"      cropping bones to {crop_r:.0f} mm sphere about the joint")
    else:
        print(f"      no crop (column fits the grid)")
    for n, m in meshes.items():
        m.apply_translation(-center)
        if crop_r:
            m = _voxel_crop_sphere(m, crop_r)
        m.export(str(out_dir / n))
    return center


# ─────────────────────────────────────────────────────────────────────────────
# VOXEL-BASED GEOMETRY HELPERS (bone cropping + soft-tissue synthesis)
# All routes go through a filled voxel grid + marching cubes, so they need
# scikit-image (imported lazily by trimesh's marching_cubes) and scipy.ndimage.
# ─────────────────────────────────────────────────────────────────────────────

def _voxelgrid_to_mesh(matrix, transform):
    """Marching-cubes a boolean voxel matrix back to a world-space Trimesh."""
    mesh = trimesh.voxel.VoxelGrid(matrix, transform=transform).marching_cubes
    mesh.apply_transform(transform)   # marching_cubes returns index space
    return mesh


def _voxel_crop_sphere(mesh, radius_mm, pitch=1.0, center=(0, 0, 0)):
    """Crop a mesh to a sphere about `center` via the voxel route (watertight,
    no polygon-triangulation engine needed)."""
    vg = mesh.voxelized(pitch=pitch).fill()
    mat = vg.matrix.copy()
    T = vg.transform
    idx = np.argwhere(mat)
    world = (T[:3, :3] @ idx.T).T + T[:3, 3]
    far = np.linalg.norm(world - np.asarray(center, float), axis=1) >= radius_mm
    f = idx[far]
    mat[f[:, 0], f[:, 1], f[:, 2]] = False
    return _voxelgrid_to_mesh(mat, T)


def _cartilage_shell(bone_mesh, thickness_mm, region_radius_mm,
                     center=(0, 0, 0), x_side=None, pitch=1.0):
    """Thin cartilage shell offset outward from the joint-facing bone surface,
    restricted to a sphere about `center` (and optionally one side in X)."""
    vg = bone_mesh.voxelized(pitch=pitch).fill()
    bone = vg.matrix
    T = vg.transform
    r = max(1, int(round(thickness_mm / pitch)))
    shell = ndimage.binary_dilation(bone, iterations=r) & ~bone
    idx = np.argwhere(shell)
    world = (T[:3, :3] @ idx.T).T + T[:3, 3]
    c = np.asarray(center, float)
    keep = np.linalg.norm(world - c, axis=1) < region_radius_mm
    if x_side == "lat":
        keep &= world[:, 0] > c[0]
    elif x_side == "med":
        keep &= world[:, 0] < c[0]
    sel = idx[keep]
    mat = np.zeros_like(shell)
    mat[sel[:, 0], sel[:, 1], sel[:, 2]] = True
    return _voxelgrid_to_mesh(mat, T)


def _rot_z_to(v):
    """4x4 rotation mapping +Z onto unit vector v."""
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, v)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return (np.eye(4) if v[2] > 0
                else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    axis /= s
    ang = np.arccos(np.clip(np.dot(z, v), -1.0, 1.0))
    return trimesh.transformations.rotation_matrix(ang, axis)


def _ring(axis_vec, center, major_radius, minor_radius):
    """Fibrocartilage torus with its axis along `axis_vec`, centred at `center`."""
    t = trimesh.creation.torus(major_radius, minor_radius)
    t.apply_transform(_rot_z_to(axis_vec))
    t.apply_translation(np.asarray(center, float))
    return t


def _head_center(bone_mesh, region_radius_mm, center=(0, 0, 0), min_pts=200):
    """Centroid of a bone's articular head — vertices near `center`, with a
    nearest-N fallback when the head is offset from the joint origin (e.g. the
    radial head relative to the humero-ulnar articulation)."""
    c = np.asarray(center, float)
    v = bone_mesh.vertices
    d = np.linalg.norm(v - c, axis=1)
    near = v[d < region_radius_mm]
    if len(near) < min_pts:
        near = v[np.argsort(d)[:min_pts]]
    return near.mean(axis=0)


def _long_axis(bone_mesh):
    """First principal axis (long direction) of a bone."""
    v = bone_mesh.vertices - bone_mesh.vertices.mean(axis=0)
    _, _, Vt = np.linalg.svd(v, full_matrices=False)
    return Vt[0]


def synthesize_soft_tissue(joint, out_dir):
    """Replace soft-tissue placeholder cubes with cartilage shells / rings
    synthesised from the joint's (already reframed, recentred, cropped) bones."""
    specs = SOFT_TISSUE_SYNTHESIS.get(joint)
    if not specs:
        return []
    out_dir = Path(out_dir)
    bones = {}
    synthesised = []

    def bone(name):
        if name not in bones:
            bones[name] = load_mesh(out_dir / name)
        return bones[name]

    for out_name, spec in specs.items():
        try:
            b = bone(spec["bone"])
            if spec.get("center") == "head":
                ctr = _head_center(b, spec.get("head_radius", 15))
            else:
                ctr = np.zeros(3)
            if spec["kind"] == "shell":
                mesh = _cartilage_shell(b, spec["thickness"], spec["radius"],
                                        center=ctr, x_side=spec.get("x_side"))
            elif spec["kind"] == "ring":
                if spec.get("axis") == "toward_head":
                    axis = _head_center(bone(spec["head_bone"]), spec["head_radius"])
                else:  # "long"
                    axis = _long_axis(b)
                mesh = _ring(axis, ctr, spec["major"], spec["minor"])
            else:
                continue
            mesh.export(str(out_dir / out_name))
            synthesised.append(out_name)
            tag = "watertight" if mesh.is_watertight else "NOT watertight"
            print(f"    Synthesised {out_name}  ({len(mesh.faces):,} faces, "
                  f"{mesh.volume:.0f} mm³)  [{tag}]")
        except Exception as e:
            print(f"    ⚠  Could not synthesise {out_name} ({e}) — placeholder kept")
    return synthesised


def fma_to_stl_path(bp3d_dir, fma_id):
    """Return path to an STL in the GitHub mirror, or None if absent."""
    p = Path(bp3d_dir) / "assets" / "BodyParts3D_data" / "stl" / f"{fma_id}.stl"
    return p if p.exists() else None


def download_dbcls_zip(cache_dir):
    """Download the DBCLS OBJ zip if not already cached. Returns path to zip."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "isa_BP3D_4.0_obj_99.zip"
    if zip_path.exists():
        print(f"  Using cached DBCLS zip: {zip_path}")
        return zip_path

    print(f"  Downloading DBCLS OBJ zip (~136 MB) ...")
    print(f"  URL: {DBCLS_ZIP_URL}")

    def progress(count, block_size, total_size):
        pct = min(100, int(count * block_size * 100 / total_size))
        bar = "#" * (pct // 5) + " " * (20 - pct // 5)
        print(f"\r    [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(DBCLS_ZIP_URL, zip_path, reporthook=progress)
    print()
    print(f"  Download complete: {zip_path}")
    return zip_path


def extract_obj_from_zip(zip_path, bp_code, out_obj_path):
    """Extract a single OBJ file from the DBCLS zip by BP code."""
    target = f"{bp_code}.obj"
    with zipfile.ZipFile(zip_path, "r") as zf:
        # The zip may contain files in a subdirectory; find the match
        matches = [n for n in zf.namelist() if n.endswith(target)]
        if not matches:
            print(f"    ❌  {target} not found in DBCLS zip")
            return False
        with zf.open(matches[0]) as src, open(out_obj_path, "wb") as dst:
            dst.write(src.read())
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CORE EXTRACTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_mesh_for_structure(struct_key, bp3d_dir, dbcls_zip_path, tmp_dir):
    """
    Locate or download source geometry for a structure key.
    Returns path to an STL or OBJ file ready for loading, or None.
    """
    fma_id, bp_code, github_available = STRUCTURES[struct_key]

    # Strategy A: GitHub STL mirror
    stl_path = fma_to_stl_path(bp3d_dir, fma_id)
    if stl_path:
        print(f"    Source: GitHub mirror  ({fma_id}.stl)")
        return stl_path, "stl"

    # Strategy B: DBCLS OBJ zip
    obj_path = Path(tmp_dir) / f"{bp_code}.obj"
    if not obj_path.exists():
        if dbcls_zip_path is None:
            print(f"    ❌  {struct_key}: not in GitHub mirror and DBCLS zip not downloaded")
            return None, None
        print(f"    Source: DBCLS zip  ({bp_code}.obj)")
        if not extract_obj_from_zip(dbcls_zip_path, bp_code, obj_path):
            return None, None
    else:
        print(f"    Source: DBCLS zip (cached)  ({bp_code}.obj)")

    return obj_path, "obj"


def process_joint(joint, side, bp3d_dir, out_dir, dbcls_zip_path, tmp_dir):
    """Extract all files for one joint and write to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_map = JOINT_FILES[joint]
    results = {"ok": [], "placeholder": [], "failed": []}

    print(f"\n  Processing joint: {joint.upper()}  (side={side})")
    print(f"  Output directory: {out_dir}")
    print()

    for stl_name, base_key in file_map.items():
        print(f"  [{stl_name}]")
        out_path = out_dir / stl_name

        # Files with None base_key are cartilage/soft-tissue — create placeholders
        if base_key is None:
            if not out_path.exists():
                _write_placeholder_stl(out_path)
                print(f"    Created minimal placeholder STL (cartilage/soft tissue)")
                print(f"    ⚠  Replace with segmented geometry when available")
            else:
                print(f"    Already exists — skipping")
            results["placeholder"].append(stl_name)
            continue

        # Resolve left/right
        struct_key = resolve_key(base_key, side)
        if struct_key not in STRUCTURES:
            # Try without side (sacrum, discs have no side)
            struct_key = base_key
        if struct_key not in STRUCTURES:
            print(f"    ❌  No mapping for '{struct_key}' — skipping")
            results["failed"].append(stl_name)
            continue

        src_path, src_type = get_mesh_for_structure(
            struct_key, bp3d_dir, dbcls_zip_path, tmp_dir
        )
        if src_path is None:
            results["failed"].append(stl_name)
            continue

        # Load, repair, save
        try:
            if src_type == "stl":
                mesh = load_mesh(src_path)
                repair_and_save(mesh, out_path)
            else:
                obj_to_stl(src_path, out_path)
            results["ok"].append(stl_name)
        except Exception as e:
            print(f"    ❌  Error processing {stl_name}: {e}")
            results["failed"].append(stl_name)

    # Reframe bones to pipeline axes, recentre on the articulation, and crop to
    # the joint region so the meshes sit centred inside the simulation grid.
    if results["ok"]:
        try:
            orient_and_recenter_joint(joint, out_dir, file_map)
        except Exception as e:
            print(f"  ⚠  Joint reframing skipped ({e}) — meshes left in world coords")
        # Synthesise cartilage / labrum / ligament targets from the bones,
        # replacing the soft-tissue placeholder cubes where configured.
        try:
            made = synthesize_soft_tissue(joint, out_dir)
            # Reclassify synthesised files out of the placeholder bucket.
            results["synthesized"] = [n for n in results["placeholder"] if n in made]
            results["placeholder"] = [n for n in results["placeholder"] if n not in made]
        except Exception as e:
            print(f"  ⚠  Soft-tissue synthesis skipped ({e}) — placeholders kept")

    # Summary
    print(f"\n  ── Summary for {joint} ──────────────────────────────────")
    print(f"  ✅  Extracted:    {len(results['ok'])} bone files")
    if results.get("synthesized"):
        print(f"  🧩  Synthesised:  {len(results['synthesized'])} soft-tissue targets "
              f"(cartilage/labrum/ligament from bone geometry)")
    if results["placeholder"]:
        print(f"  ⚠   Placeholder: {len(results['placeholder'])} files (cube — no geometry source)")
    if results["failed"]:
        print(f"  ❌  Failed:      {len(results['failed'])} files: {results['failed']}")
    return results


def _write_placeholder_stl(out_path):
    """
    Write a minimal single-triangle STL placeholder.
    The simulation pipeline skips missing/empty meshes gracefully,
    but having a valid file prevents FileNotFoundError on load.
    Replace with real segmented geometry before running simulations.
    """
    # 4x4x4 mm cube centred at origin — small enough to not affect wrapping
    box = trimesh.creation.box(extents=[4, 4, 4])
    box.export(str(out_path))


# ─────────────────────────────────────────────────────────────────────────────
# WATERTIGHTNESS REPORT
# ─────────────────────────────────────────────────────────────────────────────

def check_output_dir(out_dir):
    """Print a watertightness and face-count report for all STLs in a directory."""
    out_dir = Path(out_dir)
    stls = sorted(out_dir.glob("*.stl"))
    if not stls:
        print("  No STL files found.")
        return

    print(f"\n  {'File':<40} {'Watertight':^12} {'Faces':>10}")
    print("  " + "-" * 66)
    all_ok = True
    for stl in stls:
        try:
            mesh = trimesh.load(str(stl), force="mesh")
            wt   = "✅ yes" if mesh.is_watertight else "❌ NO"
            if not mesh.is_watertight:
                all_ok = False
            print(f"  {stl.name:<40} {wt:^12} {len(mesh.faces):>10,}")
        except Exception as e:
            print(f"  {stl.name:<40} {'ERROR':^12}  {e}")
            all_ok = False

    if all_ok:
        print("\n  All meshes are watertight — ready for simulation.")
    else:
        print("\n  ⚠  Some meshes are not watertight.")
        print("     Run the simulation script anyway — it attempts repair on load.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Windows consoles default to cp1252, which cannot encode the status
    # symbols (⚠ ✅ ❌) used below. Force UTF-8 with a safe fallback so the
    # script runs identically on Windows and POSIX terminals.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Extract BodyParts3D geometry and rename for PBM MC simulation repos."
    )
    parser.add_argument(
        "--bp3d_dir", required=True,
        help="Path to your local clone of Kevin-Mattheus-Moerman/BodyParts3D"
    )
    parser.add_argument(
        "--joint", required=True, choices=["shoulder", "elbow", "lower_back", "all"],
        help="Which joint(s) to extract"
    )
    parser.add_argument(
        "--side", default="right", choices=["right", "left"],
        help="Body side to use (default: right)"
    )
    parser.add_argument(
        "--out_dir", default=None,
        help=(
            "Output directory for STL files. "
            "Default: Raw_Mesh_Files_SHO001/, Raw_Mesh_Files_ELB001/, "
            "or Raw_Mesh_Files_LBK001/ in the current directory."
        )
    )
    parser.add_argument(
        "--cache_dir", default="./bp3d_cache",
        help="Directory to cache the DBCLS OBJ zip download (default: ./bp3d_cache)"
    )
    parser.add_argument(
        "--no_download", action="store_true",
        help="Do not download from DBCLS; only use GitHub mirror STLs"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="After extraction, print a watertightness report"
    )
    args = parser.parse_args()

    bp3d_dir = Path(args.bp3d_dir)
    if not bp3d_dir.exists():
        sys.exit(f"ERROR: --bp3d_dir '{bp3d_dir}' does not exist.")

    stl_dir = bp3d_dir / "assets" / "BodyParts3D_data" / "stl"
    if not stl_dir.exists():
        sys.exit(
            f"ERROR: Expected STL directory not found at '{stl_dir}'.\n"
            f"       Check that --bp3d_dir points to the root of the cloned repo."
        )

    joints = ["shoulder", "elbow", "lower_back"] if args.joint == "all" else [args.joint]

    DEFAULT_OUT = {
        "shoulder":   "Raw_Mesh_Files_SHO001",
        "elbow":      "Raw_Mesh_Files_ELB001",
        "lower_back": "Raw_Mesh_Files_LBK001",
    }

    # Check whether we need the DBCLS zip
    needs_download = False
    if not args.no_download:
        for joint in joints:
            for stl_name, base_key in JOINT_FILES[joint].items():
                if base_key is None:
                    continue
                struct_key = resolve_key(base_key, args.side)
                if struct_key not in STRUCTURES:
                    struct_key = base_key
                if struct_key in STRUCTURES:
                    fma_id, bp_code, github_ok = STRUCTURES[struct_key]
                    if not fma_to_stl_path(bp3d_dir, fma_id):
                        needs_download = True
                        break
            if needs_download:
                break

    dbcls_zip_path = None
    if needs_download:
        print("Some structures are not in the GitHub STL mirror.")
        print("They will be sourced from the DBCLS OBJ zip (auto-download).\n")
        dbcls_zip_path = download_dbcls_zip(args.cache_dir)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for joint in joints:
            out_dir = Path(args.out_dir) if args.out_dir else Path(DEFAULT_OUT[joint])
            process_joint(
                joint=joint,
                side=args.side,
                bp3d_dir=bp3d_dir,
                out_dir=out_dir,
                dbcls_zip_path=dbcls_zip_path,
                tmp_dir=tmp_dir,
            )
            if args.check:
                check_output_dir(out_dir)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Review the STL files visually (optional but recommended):")
    print("       pip install vedo")
    print("       python -c \"import vedo; vedo.show([vedo.load(f) for f in ")
    print("           __import__('glob').glob('Raw_Mesh_Files_*/*.stl')], axes=1)\"")
    print()
    print("  2. Add subject ID to SUBJECT_IDS in the simulation script.")
    print()
    print("  3. Run the simulation:")
    print("       python \"SHO Models_MC Results_808nm.py\"   # shoulder")
    print("       python \"ELB Models_MC Results_808nm.py\"   # elbow")
    print("       python \"LBK Models_MC Results_808nm.py\"   # lower back")
    print()
    print("  4. Shoulder/elbow cartilage, labrum, and the annular ligament are")
    print("     SYNTHESISED from the bone surfaces (offset shells + rings), since")
    print("     BodyParts3D has no articular soft tissue. They are modelling")
    print("     approximations for MC fluence targets — replace with segmented")
    print("     MRI (shoulder/elbow atlas) for measured accuracy. Any remaining")
    print("     placeholder cubes (e.g. lower-back discs) have no geometry source.")
    print()
    print("  Attribution: BodyParts3D, (c) DBCLS, CC-BY-SA 2.1 Japan")
    print("  Cite: Mitsuhashi N et al., Nucleic Acids Res. 2009;37:D782-5")


if __name__ == "__main__":
    main()
