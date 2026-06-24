# PhiSat-2 MajorTOM Tools

Small utilities to build a MajorTOM-like PhiSat-2 L1C package.

The goal is to represent PhiSat-2 L1C acquisitions as an additional modality on the official MajorTOM cell grid. Each generated sample corresponds to one MajorTOM cell, with all 8 PhiSat-2 bands warped/resampled to the official MajorTOM CRS, geotransform, resolution and shape.

## Current scope

This repository currently focuses on the packaging step:

1. read raw PhiSat-2 L1C acquisitions;
2. use precomputed corrected georeferencing / alignment metadata;
3. warp all 8 PhiSat-2 bands to selected MajorTOM cells;
4. write a MajorTOM-like package with:
   - `index/global.parquet`
   - `images/part_00001.parquet`
   - `metadata/manifest.csv`
   - `metadata/dataset_summary.json`
   - `README.md`

The generation of the corrected PhiSat-2 / Sentinel-2 LightGlue triplets is not included yet. It will be documented separately.

## Output structure

```text
phisat2_l1c_major_tom_demo/
├── index/
│   └── global.parquet
├── images/
│   └── part_00001.parquet
├── metadata/
│   ├── manifest.csv
│   └── dataset_summary.json
└── README.md
Image parquet schema

images/part_00001.parquet contains one row per sample.

Columns:

product_id
grid_cell
product_datetime
thumbnail
valid_mask
B00
B01
B02
B03
B04
B05
B06
B07

The band columns are GeoTIFF bytes. Each band is stored as a single-band GeoTIFF with the official MajorTOM cell CRS, transform, resolution and shape.

Index schema

index/global.parquet contains one row per generated PhiSat-2/MajorTOM sample.

Columns:

id
sample_id
grid_cell
product_id
processing_level
date
majortom:code_100km
majortom:code_1000km
majortom:mgrs_tile
crs
ul_x
ul_y
width
height
resolution_m
geotransform
bbox
geometry
admin:country
admin:state
admin:district
phisat2:valid_fraction
phisat2:source_product_name
phisat2:pair
phisat2:alignment_source
phisat2:alignment_method
phisat2:output_grid
Band mapping
Output band	Wavelength
B00	625 nm
B01	490 nm
B02	560 nm
B03	665 nm
B04	705 nm
B05	740 nm
B06	783 nm
B07	842 nm
Alignment

This repository assumes that corrected georeferencing is already available.

The demo used georeferencing derived from LightGlue-georeferenced PhiSat-2 / Sentinel-2 patches:

https://huggingface.co/datasets/lorenzopapa53/phisat2-s2-lightglue-triplets

The current alignment method is recorded as:

full_scene_affine_from_lightglue_patch_georeferencing_with_horizontal_flip
Quick start

In the ESA/NEOHPC environment, use phipy:

phipy scripts/build_demo_package.py
phipy scripts/inspect_package.py
phipy scripts/audit_package.py
Status

Prototype / demo repository. The full triplet/georeferencing generation pipeline is not included yet.
