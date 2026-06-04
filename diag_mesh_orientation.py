import numpy as np
import os

def load_stl_vertices(path):
    import trimesh
    mesh = trimesh.load(path, force='mesh')
    return np.array(mesh.vertices)

subjects = ["OKS001", "OKS002", "OKS003", "OKS004", "OKS006", "OKS007", "OKS008", "OKS009"]
base = "C:/Users/chris/PycharmProjects/PythonProject/.venv/Scripts"

results = {}

for subj in subjects:
    subj_dir = f"{base}/Raw_Mesh_Files_{subj}"
    femur_path = f"{subj_dir}/femur_raw.stl"
    tibia_path = f"{subj_dir}/tibia_raw.stl"
    cart_path  = f"{subj_dir}/femur_cartilage_raw.stl"

    has_femur = os.path.exists(femur_path)
    has_tibia = os.path.exists(tibia_path)
    has_cart  = os.path.exists(cart_path)

    if not has_femur and not has_tibia:
        results[subj] = None
        print(f"{subj}: MISSING both femur and tibia — skipping")
        continue

    meshes = {}
    if has_femur:
        print(f"  Loading {subj} femur...")
        meshes['femur'] = load_stl_vertices(femur_path)
    if has_tibia:
        print(f"  Loading {subj} tibia...")
        meshes['tibia'] = load_stl_vertices(tibia_path)
    if has_cart:
        print(f"  Loading {subj} femur_cartilage...")
        meshes['femur_cart'] = load_stl_vertices(cart_path)

    # Per-mesh stats
    stats = {}
    for name, verts in meshes.items():
        centroid = verts.mean(axis=0)
        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
        bb_center = (bb_min + bb_max) / 2.0
        bb_dims = bb_max - bb_min
        stats[name] = {
            'centroid': centroid,
            'bb_min': bb_min,
            'bb_max': bb_max,
            'bb_center': bb_center,
            'bb_dims': bb_dims,
            'n_verts': len(verts)
        }

    # Combined bounding box
    all_verts = np.vstack(list(meshes.values()))
    combined_bb_min = all_verts.min(axis=0)
    combined_bb_max = all_verts.max(axis=0)
    combined_bb_center = (combined_bb_min + combined_bb_max) / 2.0
    combined_bb_dims = combined_bb_max - combined_bb_min

    # PCA on femur vertices
    pca_info = None
    if 'femur' in meshes:
        fv = meshes['femur']
        fv_centered = fv - fv.mean(axis=0)
        cov = np.cov(fv_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_vec = eigenvectors[:, -1]
        pca_info = {
            'eigenvalues': eigenvalues,
            'principal_vec': principal_vec,
            'eigenvectors': eigenvectors
        }

    femur_centroid = stats['femur']['centroid'] if 'femur' in stats else None
    tibia_centroid = stats['tibia']['centroid'] if 'tibia' in stats else None

    z_indicator = None
    x_indicator = None
    y_indicator = None
    if femur_centroid is not None and tibia_centroid is not None:
        z_indicator = femur_centroid[2] - tibia_centroid[2]
        x_indicator = femur_centroid[0] - tibia_centroid[0]
        y_indicator = femur_centroid[1] - tibia_centroid[1]

    results[subj] = {
        'stats': stats,
        'combined_bb_center': combined_bb_center,
        'combined_bb_dims': combined_bb_dims,
        'pca_info': pca_info,
        'z_indicator': z_indicator,
        'x_indicator': x_indicator,
        'y_indicator': y_indicator,
        'has_femur': has_femur,
        'has_tibia': has_tibia,
        'has_cart': has_cart
    }

print("\n" + "="*110)
print("KNEE MESH ORIENTATION DIAGNOSTIC — SUMMARY TABLE")
print("="*110)
print("  Femur/Tibia centroids relative to combined bbox center")
print("  dX/dY/dZ = femur_centroid minus tibia_centroid (raw world coords)")
print("-"*110)
header = f"{'Subj':<8} {'F_cx':>8} {'F_cy':>8} {'F_cz':>8} | {'T_cx':>8} {'T_cy':>8} {'T_cz':>8} | {'dX(F-T)':>9} {'dY(F-T)':>9} {'dZ(F-T)':>9}"
print(header)
print("-"*110)

for subj in subjects:
    r = results.get(subj)
    if r is None:
        print(f"{subj:<8}  MISSING")
        continue
    cbc = r['combined_bb_center']
    stats = r['stats']
    fc = stats['femur']['centroid'] - cbc if 'femur' in stats else np.array([float('nan')]*3)
    tc = stats['tibia']['centroid'] - cbc if 'tibia' in stats else np.array([float('nan')]*3)
    xi = r['x_indicator'] if r['x_indicator'] is not None else float('nan')
    yi = r['y_indicator'] if r['y_indicator'] is not None else float('nan')
    zi = r['z_indicator'] if r['z_indicator'] is not None else float('nan')
    marker = "  <<< CHECK" if r['z_indicator'] is not None and abs(r['z_indicator']) < 5 else ""
    print(f"{subj:<8} {fc[0]:>8.1f} {fc[1]:>8.1f} {fc[2]:>8.1f} | {tc[0]:>8.1f} {tc[1]:>8.1f} {tc[2]:>8.1f} | {xi:>9.1f} {yi:>9.1f} {zi:>9.1f}{marker}")

print()
print("="*110)
print("BOUNDING BOX DIMENSIONS  (femur and combined, presumed mm)")
print("="*110)
header2 = f"{'Subj':<8} {'F_Xdim':>8} {'F_Ydim':>8} {'F_Zdim':>8} | {'Comb_X':>8} {'Comb_Y':>8} {'Comb_Z':>8}"
print(header2)
print("-"*110)
for subj in subjects:
    r = results.get(subj)
    if r is None:
        print(f"{subj:<8}  MISSING")
        continue
    stats = r['stats']
    fd = stats['femur']['bb_dims'] if 'femur' in stats else np.array([float('nan')]*3)
    cd = r['combined_bb_dims']
    print(f"{subj:<8} {fd[0]:>8.1f} {fd[1]:>8.1f} {fd[2]:>8.1f} | {cd[0]:>8.1f} {cd[1]:>8.1f} {cd[2]:>8.1f}")

print()
print("="*110)
print("FEMUR ABSOLUTE BBOX (world coordinates, to detect axis orientation)")
print("="*110)
header5 = f"{'Subj':<8} {'F_xmin':>9} {'F_xmax':>9} {'F_ymin':>9} {'F_ymax':>9} {'F_zmin':>9} {'F_zmax':>9}"
print(header5)
print("-"*110)
for subj in subjects:
    r = results.get(subj)
    if r is None:
        print(f"{subj:<8}  MISSING")
        continue
    stats = r['stats']
    if 'femur' not in stats:
        print(f"{subj:<8}  no femur")
        continue
    fs = stats['femur']
    print(f"{subj:<8} {fs['bb_min'][0]:>9.1f} {fs['bb_max'][0]:>9.1f} {fs['bb_min'][1]:>9.1f} {fs['bb_max'][1]:>9.1f} {fs['bb_min'][2]:>9.1f} {fs['bb_max'][2]:>9.1f}")

print()
print("="*110)
print("PCA — FEMUR PRINCIPAL EIGENVECTOR (largest eigenvalue = primary long axis)")
print("  Consistent sign across subjects = same orientation")
print("  Sign flip in one subject = mirrored/reflected coordinate system")
print("="*110)
header3 = f"{'Subj':<8} {'EV1_x':>10} {'EV1_y':>10} {'EV1_z':>10} | {'EV2_x':>10} {'EV2_y':>10} {'EV2_z':>10} | {'EV3_x':>10} {'EV3_y':>10} {'EV3_z':>10}"
print(header3)
print(f"{'':8} {'(smallest)':>32} {'(mid)':>32} {'(largest/primary)':>32}")
print("-"*110)
for subj in subjects:
    r = results.get(subj)
    if r is None:
        print(f"{subj:<8}  MISSING")
        continue
    pca = r['pca_info']
    if pca is None:
        print(f"{subj:<8}  no femur")
        continue
    evecs = pca['eigenvectors']
    evals = pca['eigenvalues']
    # columns of evecs are eigenvectors; eigh returns ascending, so col 0=smallest, col 2=largest
    v1 = evecs[:, 0]
    v2 = evecs[:, 1]
    v3 = evecs[:, 2]
    print(f"{subj:<8} {v1[0]:>10.4f} {v1[1]:>10.4f} {v1[2]:>10.4f} | {v2[0]:>10.4f} {v2[1]:>10.4f} {v2[2]:>10.4f} | {v3[0]:>10.4f} {v3[1]:>10.4f} {v3[2]:>10.4f}")

print()
print("="*110)
print("EIGENVALUES (ascending)")
print("="*110)
header4b = f"{'Subj':<8} {'Eval_sm':>14} {'Eval_mid':>14} {'Eval_lg':>14}  (ratio lg/sm indicates elongation)"
print(header4b)
print("-"*110)
for subj in subjects:
    r = results.get(subj)
    if r is None or r['pca_info'] is None:
        continue
    ev = r['pca_info']['eigenvalues']
    ratio = ev[2]/ev[0] if ev[0] > 0 else float('nan')
    print(f"{subj:<8} {ev[0]:>14.1f} {ev[1]:>14.1f} {ev[2]:>14.1f}  ratio={ratio:.1f}")

print()
print("="*110)
print("ABSOLUTE CENTROIDS (raw world coordinates)")
print("="*110)
header4 = f"{'Subj':<8} {'F_x':>9} {'F_y':>9} {'F_z':>9} | {'T_x':>9} {'T_y':>9} {'T_z':>9} | {'CombCtr_x':>10} {'CombCtr_y':>10} {'CombCtr_z':>10}"
print(header4)
print("-"*110)
for subj in subjects:
    r = results.get(subj)
    if r is None:
        print(f"{subj:<8}  MISSING")
        continue
    stats = r['stats']
    fc = stats['femur']['centroid'] if 'femur' in stats else np.array([float('nan')]*3)
    tc = stats['tibia']['centroid'] if 'tibia' in stats else np.array([float('nan')]*3)
    cbc = r['combined_bb_center']
    print(f"{subj:<8} {fc[0]:>9.1f} {fc[1]:>9.1f} {fc[2]:>9.1f} | {tc[0]:>9.1f} {tc[1]:>9.1f} {tc[2]:>9.1f} | {cbc[0]:>10.1f} {cbc[1]:>10.1f} {cbc[2]:>10.1f}")

print()
print("="*110)
print("INTERPRETATION GUIDE")
print("  dZ(F-T) > 0  => femur centroid ABOVE tibia in Z    (anatomically standard)")
print("  dZ(F-T) < 0  => femur centroid BELOW tibia in Z    (Z-axis FLIPPED)")
print("  dZ(F-T) ~ 0  => femur and tibia at same Z level    (unusual — check data)")
print("  Consistent bbox dims across subjects = same scale/units")
print("  PCA principal eigenvector sign flip in one subject = reflected mesh")
print("="*110)
