from pathlib import Path
from io import BytesIO
import tarfile
import json
import pandas as pd
import rasterio

TAR_PATH = Path("phisat2_l1c_major_tom_demo.tar.gz")
ROOT = Path("phisat2_l1c_major_tom_demo")

EXPECTED_INDEX_COLS = [
    "id",
    "sample_id",
    "grid_cell",
    "product_id",
    "processing_level",
    "date",
    "majortom:code_100km",
    "majortom:code_1000km",
    "majortom:mgrs_tile",
    "crs",
    "ul_x",
    "ul_y",
    "width",
    "height",
    "resolution_m",
    "geotransform",
    "bbox",
    "geometry",
    "admin:country",
    "admin:state",
    "admin:district",
    "phisat2:valid_fraction",
    "phisat2:source_product_name",
    "phisat2:pair",
    "phisat2:alignment_source",
    "phisat2:alignment_method",
    "phisat2:output_grid",
]

EXPECTED_IMAGE_COLS = [
    "product_id",
    "grid_cell",
    "product_datetime",
    "thumbnail",
    "valid_mask",
    "B00",
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
]

BANDS = ["B00", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "valid_mask"]


def fail(msg):
    raise RuntimeError("AUDIT FAILED: " + msg)


print("=== 1. TAR archive check ===")
if not TAR_PATH.exists():
    fail(f"Archive not found: {TAR_PATH}")

with tarfile.open(TAR_PATH, "r:gz") as tar:
    names = tar.getnames()
    print("Archive members:", len(names))
    required = [
        f"{ROOT}/index/global.parquet",
        f"{ROOT}/images/part_00001.parquet",
        f"{ROOT}/metadata/manifest.csv",
        f"{ROOT}/metadata/dataset_summary.json",
        f"{ROOT}/README.md",
    ]
    for r in required:
        if r not in names:
            fail(f"Missing in archive: {r}")
    suspicious = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
    if suspicious:
        fail(f"Suspicious paths in archive: {suspicious[:5]}")
    print("OK archive structure")


print("\n=== 2. Local folder files check ===")
required_paths = [
    ROOT / "index" / "global.parquet",
    ROOT / "images" / "part_00001.parquet",
    ROOT / "metadata" / "manifest.csv",
    ROOT / "metadata" / "dataset_summary.json",
    ROOT / "README.md",
]
for p in required_paths:
    if not p.exists():
        fail(f"Missing local file: {p}")
    print(p, p.stat().st_size, "bytes")


print("\n=== 3. Read parquet/csv/json ===")
idx = pd.read_parquet(ROOT / "index" / "global.parquet")
img = pd.read_parquet(ROOT / "images" / "part_00001.parquet")
manifest = pd.read_csv(ROOT / "metadata" / "manifest.csv")
summary = json.loads((ROOT / "metadata" / "dataset_summary.json").read_text())

print("index shape:", idx.shape)
print("image shape:", img.shape)
print("manifest shape:", manifest.shape)

if idx.shape[0] != 5:
    fail(f"Index should have 5 rows, got {idx.shape[0]}")
if img.shape[0] != 5:
    fail(f"Images should have 5 rows, got {img.shape[0]}")
if manifest.shape[0] != 5:
    fail(f"Manifest should have 5 rows, got {manifest.shape[0]}")


print("\n=== 4. Column checks ===")
if idx.columns.tolist() != EXPECTED_INDEX_COLS:
    print("Expected index cols:", EXPECTED_INDEX_COLS)
    print("Actual index cols:", idx.columns.tolist())
    fail("Index columns do not match expected compact schema")

if img.columns.tolist() != EXPECTED_IMAGE_COLS:
    print("Expected image cols:", EXPECTED_IMAGE_COLS)
    print("Actual image cols:", img.columns.tolist())
    fail("Image columns do not match expected schema")

print("OK columns")


print("\n=== 5. Forbidden/debug column checks ===")
forbidden = [
    "phisat2:source_path",
    "phisat2:s2_path",
    "phisat2:out_png",
    "phisat2:out_phi_tif",
    "terrain:elevation",
    "socio:gdp",
    "soil:clay",
    "climate:temperature",
]
bad = [c for c in forbidden if c in idx.columns]
if bad:
    fail(f"Forbidden/debug columns still present: {bad}")
print("OK no debug/heavy columns")


print("\n=== 6. ID consistency checks ===")
idx_ids = set(idx["id"])
manifest_ids = set(manifest["major_tom_id"])
if idx_ids != manifest_ids:
    fail("Index id and manifest major_tom_id mismatch")

for _, r in idx.iterrows():
    sid = r["sample_id"]
    pid = r["product_id"]
    cid = r["id"]

    if not sid.startswith(cid + "_PHISAT2_L1C_"):
        fail(f"Bad sample_id prefix: {sid}")

    if not str(pid).startswith("PHISAT2_L1C_"):
        fail(f"Bad product_id: {pid}")

    if len(str(pid).split("_")[-1]) != 9:
        fail(f"Product id is not 9-digit padded: {pid}")

    if r["processing_level"] != "L1C":
        fail(f"Unexpected processing_level: {r['processing_level']}")

    if float(r["resolution_m"]) != 10.0:
        fail(f"Unexpected resolution_m: {r['resolution_m']}")

    if int(r["width"]) != 1068 or int(r["height"]) != 1068:
        fail(f"Unexpected shape in index: {r['width']} x {r['height']}")

    if not (0.0 <= float(r["phisat2:valid_fraction"]) <= 1.0):
        fail(f"Invalid valid_fraction: {r['phisat2:valid_fraction']}")

    if not str(r["phisat2:source_product_name"]).startswith("PHISAT-2_L1_"):
        fail(f"Bad source product name: {r['phisat2:source_product_name']}")

print("OK IDs")


print("\n=== 7. Image rows and GeoTIFF bytes checks ===")
for i, row in img.iterrows():
    full_cell = "MT10_" + str(row["grid_cell"])
    idx_row = idx[idx["id"] == full_cell]
    if len(idx_row) != 1:
        fail(f"Could not match image grid_cell to index id: {row['grid_cell']}")

    idx_row = idx_row.iloc[0]
    expected_crs = idx_row["crs"]
    gt = idx_row["geotransform"]
if hasattr(gt, "tolist"):
    expected_transform = tuple(float(x) for x in gt.tolist())
elif isinstance(gt, (list, tuple)):
    expected_transform = tuple(float(x) for x in gt)
else:
    import re
    nums = re.findall(r"[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?", str(gt))
    expected_transform = tuple(float(x) for x in nums)
if len(expected_transform) != 6:
    fail(f"Could not parse geotransform for {idx_row['id']}: {gt}")

    if row["product_id"] != idx_row["product_id"]:
        fail(f"Product mismatch for {full_cell}: image={row['product_id']} index={idx_row['product_id']}")

    for band in BANDS:
        data = row[band]
        if not isinstance(data, (bytes, bytearray)):
            fail(f"{band} for {full_cell} is not bytes")

        with rasterio.open(BytesIO(data)) as src:
            if src.width != 1068 or src.height != 1068:
                fail(f"{band} wrong shape for {full_cell}: {src.width}x{src.height}")
            if str(src.crs) != expected_crs:
                fail(f"{band} CRS mismatch for {full_cell}: {src.crs} vs {expected_crs}")

            tr = src.transform
            got = (tr.c, tr.a, tr.b, tr.f, tr.d, tr.e)
            # geotransform convention: [ul_x, px_w, rot_x, ul_y, rot_y, px_h]
            for a, b in zip(got, expected_transform):
                if abs(float(a) - float(b)) > 1e-6:
                    fail(f"{band} transform mismatch for {full_cell}: {got} vs {expected_transform}")

            if band == "valid_mask":
                if src.count != 1:
                    fail(f"valid_mask count should be 1 for {full_cell}")
            else:
                if src.count != 1:
                    fail(f"{band} count should be 1 for {full_cell}")
                if src.dtypes[0] != "uint16":
                    fail(f"{band} dtype should be uint16 for {full_cell}, got {src.dtypes[0]}")

    print("OK", full_cell)

print("\n=== 8. Summary check ===")
if summary.get("n_samples") != 5:
    fail(f"summary n_samples should be 5, got {summary.get('n_samples')}")

if "band_mapping" not in summary:
    fail("summary missing band_mapping")

for b in ["B00", "B01", "B02", "B03", "B04", "B05", "B06", "B07"]:
    if b not in summary["band_mapping"]:
        fail(f"summary missing band mapping for {b}")

print("OK summary")


print("\n=== AUDIT PASSED ===")
print("Archive ready to send:", TAR_PATH)
