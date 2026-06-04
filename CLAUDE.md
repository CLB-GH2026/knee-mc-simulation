# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active Scripts

All simulation scripts live in `.venv/Scripts/`:

| File | Purpose |
|---|---|
| `OKS Knee Models_MC Results_808nm.py` | Batch pipeline — runs all OKS subjects at 808 nm |
| `OKS Knee Models_MC Results_650nm.py` | Batch pipeline — runs all OKS subjects at 650 nm |
| `Knee Model_Tissue On_Off_808nm.py` | Single-subject interactive viewer at 808 nm (OKS004) |
| `Knee Model_Tissue On_Off_650nm.py` | Single-subject interactive viewer at 650 nm (OKS004) |

Run any script directly with Python from the project root:
```bash
python ".venv/Scripts/OKS Knee Models_MC Results_808nm.py"
```

## Project Overview

Monte Carlo (MC) photon transport simulation of photobiomodulation (PBM) laser delivery to the knee joint. The pipeline models how NIR/red light propagates through anatomically realistic tissue layers to reach cartilage and synovial fluid targets.

### Pipeline (per subject / per run)

1. **STL → voxel label volume** (`build_label_volume`): loads one STL mesh per tissue, ray-casts along Z to fill an integer label array. Auto-detects Z-axis inversion by comparing femur vs tibia centroid Z (`AUTO_ORIENT = True`).
2. **Synovial fluid fill** (`add_synovial_fluid`): dilates cartilage mask to fill joint space with label 14.
3. **Soft-tissue wrapping** (`add_wrapping_layers`): binary dilation adds concentric muscle → adipose → skin shells. Thickness controlled by `MUSCLE_THICK_MM`, `ADIPOSE_THICK_MM`, `SKIN_THICK_MM`.
4. **Epidermis labelling** (`add_epidermis_layer`): outermost skin voxel ring relabelled as epidermis (label 15) with melanin-condition-specific optical properties scaled by `_EPI_SCALE = 0.2/1.0`.
5. **Joint-line Z detection** (`find_joint_line_z`): finds Z slice with peak cartilage + synovial density; source Z positions are set to this offset so sources illuminate the joint line directly.
6. **Source placement** (`find_surface_source_positions`): 3 sources (1 posterior, 2 anterior) snapped to nearest epidermis voxel; direction aimed at `[0, 0, jl_z]` (joint centre, not geometric centre).
7. **pmcx simulation** (`run_pmcx`): GPU-accelerated MC. Fluence output in mW/cm² scaled by `SOURCE_POWER_MW × SOURCE_DUTY_CYCLE × SOURCE_OPT_EFF`.
8. **Analysis** (`analyze_fluence_absorption`): per-tissue mean fluence, absorbed power, and % of input power. Power reference = `SOURCE_POWER_MW × SOURCE_DUTY_CYCLE × SOURCE_OPT_EFF × n_sources`.
9. **Output**: interactive Plotly HTML overlays + CSV results table.

### Key Configuration Constants (top of each script)

```python
VOXEL_SIZE       = 1.0      # mm/voxel
GRID_DIMS_MM     = (150, 140, 285)
AUTO_ORIENT      = True     # auto-correct Z-axis inversion
MUSCLE_THICK_MM  = 12       # wrapping layer thicknesses (mm)
ADIPOSE_THICK_MM = 6
SKIN_THICK_MM    = 2        # (3 in On_Off scripts)
FLUENCE_OUTPUT   = None     # None = run pmcx; True = load saved .npy files
SOURCE_POWER_MW  = 50       # 808nm batch (120 for 650nm batch)
SOURCE_DUTY_CYCLE = 0.75
SOURCE_OPT_EFF   = 0.85
OPTIMIZE_SOURCES = False    # True = reciprocity optimizer instead of fixed positions
```

### Coordinate System

- **+Y = anterior**, −Y = posterior
- +X / −X = medial / lateral
- **+Z = superior** (femur above tibia in standard orientation)
- `world_pos` in `src_configs` is relative to `mesh_center` (bounding box midpoint)
- `jl_z` is the Z offset of the joint line from `mesh_center` (typically −10 to −20 mm)

### Tissue Labels

| Label | Tissue |
|---|---|
| 1 | Femur bone |
| 2 | Tibia bone |
| 3 | Fibula bone |
| 4 | Patella bone |
| 5 | Lateral meniscus |
| 6 | Medial meniscus |
| 7 | Femoral cartilage |
| 8 | Lateral tibial cartilage |
| 9 | Medial tibial cartilage |
| 10 | Patellar cartilage / ligament |
| 11 | Muscle |
| 12 | Adipose |
| 13 | Skin |
| 14 | Synovial fluid |
| 15 | Epidermis |

### Optical Properties (808 nm / 650 nm)

Properties follow `opt(µa, µs', g, n)` where `µs = µs'/(1−g)` for pmcx.
Epidermis properties are wavelength-specific and scaled by `_EPI_SCALE = 0.2` (physical thickness correction). Three melanin conditions: fair / olive / dark (Fitzpatrick I-II / III-IV / V-VI).

### Known Issues / Design Decisions

- **Uniform wrapping**: muscle/adipose shells are isotropic dilations — anteriorly this overestimates soft-tissue depth (patella is nearly subcutaneous). Reduce `MUSCLE_THICK_MM` to 6–8 mm to model anterior access more accurately.
- **OKS002 Z-inversion**: diagnosed as STL export artifact (LPS/RAS convention mismatch). `AUTO_ORIENT` corrects this automatically by comparing femur vs tibia Z centroids.
- **Power budget**: total absorbed / total input ≈ 25–35% at 808 nm and 35–55% at 650 nm is physically correct for NIR in tissue. Muscle + adipose dominate absorption by volume; cartilage mean fluence (mW/cm²) is the relevant PBM metric.
- **Reciprocity optimizer** (`OPTIMIZE_SOURCES = True`): places a virtual isotropic source at the joint centroid, runs a reduced-photon MC, and uses surface fluence to identify optimal external positions via greedy NMS.

## Key Installed Libraries

- **numpy, scipy** — numerical computation
- **trimesh** — STL loading and mesh repair
- **pmcx** — GPU photon Monte Carlo simulation
- **plotly** — interactive 3D HTML visualization
- **xlsxwriter, python-pptx, Pillow** — export formats

## Code Formatter

Black is configured as the formatter (`.idea/misc.xml`).
