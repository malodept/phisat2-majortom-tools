from io import BytesIO
import numpy as np
import pandas as pd
import rasterio
from PIL import Image

ROOT = "phisat2_l1c_major_tom_aligned_demo"

idx = pd.read_parquet(f"{ROOT}/index/global.parquet")
img = pd.read_parquet(f"{ROOT}/images/part_00001.parquet")
manifest = pd.read_csv(f"{ROOT}/metadata/manifest.csv")

print("INDEX SHAPE:", idx.shape)
print("IMAGE SHAPE:", img.shape)
print("MANIFEST SHAPE:", manifest.shape)

print("\nIMAGE COLUMNS:")
print(img.columns.tolist())

print("\nMANIFEST:")
print(manifest.to_string(index=False))

row = img.iloc[0]

print("\nFIRST IMAGE ROW:")
print("product_id:", row["product_id"])
print("grid_cell:", row["grid_cell"])
print("product_datetime:", row["product_datetime"])

for band in ["B00", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "valid_mask"]:
    b = row[band]
    with rasterio.open(BytesIO(b)) as src:
        arr = src.read(1)
        print(band, "shape=", arr.shape, "dtype=", arr.dtype, "crs=", src.crs, "transform=", src.transform)

thumb = Image.open(BytesIO(row["thumbnail"]))
print("thumbnail:", thumb.mode, thumb.size)

print("\nFIRST INDEX ROW SELECTED FIELDS:")
cols = [
    "id", "sample_id", "grid_cell", "product_id", "processing_level",
    "crs", "ul_x", "ul_y", "width", "height", "resolution_m",
    "phisat2:valid_fraction", "phisat2:alignment_source",
    "phisat2:geometric_model", "phisat2:orientation_correction",
]
for c in cols:
    if c in idx.columns:
        print(c, ":", idx.iloc[0][c])
