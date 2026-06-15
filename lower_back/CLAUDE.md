# CLAUDE.md — Lower Back (Lumbar) PBM Monte Carlo Simulation

This file provides guidance to Claude Code when working with this repository.
Mirrors the structure of the knee `knee-mc-simulation` repo.

## Active Scripts

| File | Purpose |
|---|---|
| `LBK Models_MC Results_808nm.py` | Batch pipeline — all subjects at 808 nm |
| `LBK Models_MC Results_650nm.py` | Batch pipeline — all subjects at 650 nm |

## Project Overview

Monte Carlo photon transport simulation of PBM laser delivery to the lumbar spine
(L3–S1 segment).  Models light propagation through the thick paraspinal muscle
stack (erector spinae + multifidus, ~35 mm) to reach intervertebral disc targets
(L4/L5 and L5/S1 prioritised for clinical relevance to lower back pain).

## Pipeline

Identical to the knee pipeline with these substitutions:
- Cartilage → intervertebral disc annulus fibrosus (fibrocartilage, labels 5–7)
- Synovial fluid → nucleus pulposus (near-water, label 14, synthesised by dilation of annulus)
- Menisci → not present
- Patella → not present

## Key Configuration Constants

```python
VOXEL_SIZE       = 1.0
GRID_DIMS_MM     = (200, 180, 280)   # tall to capture L3–S1 stack
AUTO_ORIENT      = True              # L3 above S1 check
MUSCLE_THICK_MM  = 35   # paraspinal bulk — LARGEST of all Kineon targets
ADIPOSE_THICK_MM =  8   # posterior subcutaneous fat thicker than knee
SKIN_THICK_MM    =  2
SOURCE_POWER_MW  = 50   # 808nm  (120 for 650nm)
```

## Coordinate System

- **+Z = superior** (cranial — L3 above S1)
- **+Y = anterior** (ventral)
- **+X = lateral** (right)

## Tissue Labels

| Label | Tissue |
|---|---|
| 1 | L3 vertebral body |
| 2 | L4 vertebral body (primary PBM target) |
| 3 | L5 vertebral body |
| 4 | S1 sacrum |
| 5 | L3–L4 disc annulus (fibrocartilage) |
| 6 | L4–L5 disc annulus (primary PBM target) |
| 7 | L5–S1 disc annulus |
| 11 | Muscle |
| 12 | Adipose |
| 13 | Skin |
| 14 | Nucleus pulposus (synthesised) |
| 15 | Epidermis |

## Default Source Placement

Three posterior sources flanking the spinous process (Move+ lumbar pad placement):
- Centre (0, −80 mm, jl_z)
- Left  (−30, −75 mm, jl_z)
- Right (+30, −75 mm, jl_z)

## Required STL Files Per Subject

Place in `Scripts/Raw_Mesh_Files_LBK###/`:
```
L3_raw.stl
L4_raw.stl
L5_raw.stl
S1_raw.stl
L3L4_disc_raw.stl
L4L5_disc_raw.stl
L5S1_disc_raw.stl
```

## Recommended STL Sources

| Tissue | Source | Notes |
|---|---|---|
| L3–S1 vertebrae | [BodyParts3D GitHub](https://github.com/Kevin-Mattheus-Moerman/BodyParts3D) | CC-BY-SA; individual vertebra STLs |
| Discs (annulus + nucleus) | [SpineWeb / DIKU challenge](http://spineweb.digitalimaginggroup.ca/) | MRI-segmented; best disc geometry |
| All tissues (auto) | TotalSegmentator on lumbar CT | `TotalSegmentator -i ct.nii -o seg/` |
| Vertebrae + discs | NIH 3D Print Exchange lumbar models | CT-derived; bone only |

## Key Differences from Knee Pipeline

- `MUSCLE_THICK_MM = 35` — paraspinals at L4/L5 are the deepest soft tissue of all targets
- `GRID_DIMS_MM` tall (280 mm Z) to accommodate L3–S1 vertebral stack height (~150 mm)
- Orientation uses `L3-bone` vs `S1-bone` centroids
- No cartilage labels — disc annulus replaces them (same fibrocartilage optical class)
- Nucleus pulposus synthesised from disc dilation (water-like optical props at 808/650 nm)
- Depth histogram zone references: skin/adipose ~1 cm, paraspinals ~4.5 cm, disc ~6.5 cm
