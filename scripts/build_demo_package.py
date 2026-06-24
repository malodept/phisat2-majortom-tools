from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from PIL import Image
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling


# ============================================================
# CONFIG
# ============================================================

BATCH_REPORT = Path("outputs_batch_compare_s2_vs_phisat/batch_report.csv")
CANDIDATES = Path("outputs_fullscene_coverage_from_lorenzo_affine/raw_candidates_top50_FIXED.csv")
FITS = Path("outputs_fullscene_coverage_from_lorenzo_affine/fit_reports.csv")
MAJORTOM_INDEX = Path("hf_majortom_index/global.parquet")

OUTROOT = Path("phisat2_l1c_major_tom_aligned_demo")
N_SAMPLES = 5

CELL_SIZE = 1068
TARGET_RES_M = 10.0

# Raw PhiSat-2 has 8 bands.
PHISAT_BANDS = {
    "B00": {"raw_index_1based": 1, "wavelength_nm": 625},
    "B01": {"raw_index_1based": 2, "wavelength_nm": 490},
    "B02": {"raw_index_1based": 3, "wavelength_nm": 560},
    "B03": {"raw_index_1based": 4, "wavelength_nm": 665},
    "B04": {"raw_index_1based": 5, "wavelength_nm": 705},
    "B05": {"raw_index_1based": 6, "wavelength_nm": 740},
    "B06": {"raw_index_1based": 7, "wavelength_nm": 783},
    "B07": {"raw_index_1based": 8, "wavelength_nm": 842},
}


# ============================================================
# HELPERS
# ============================================================

def cell_short(cell_id: str) -> str:
    return cell_id.replace("MT10_", "")


def product_short(product_id: int | str) -> str:
    return f"{int(product_id):09d}"


def sample_id(cell_id: str, product_id: int | str) -> str:
    return f"{cell_id}_PHISAT2_L1C_{product_short(product_id)}"


def pair_datetime_iso(pair: str) -> str:
    # pair_000001636_20250218161743_20250218161746_8762AC01
    parts = str(pair).split("_")
    if len(parts) < 3:
        return str(pair)
    s = parts[2]
    if len(s) != 14:
        return s
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}:{s[12:14]}"


def clean_gt(gt):
    s = str(gt)
    s = re.sub(r"np\.int\d+\(", "", s)
    s = re.sub(r"np\.float\d+\(", "", s)
    s = s.replace(")", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", s)
    if len(nums) < 6:
        raise ValueError(f"bad geotransform: {gt}")
    return [float(x) for x in nums[:6]]


def bbox_from_gt(gt, width=CELL_SIZE, height=CELL_SIZE):
    x0, px, _, y0, _, py = gt
    xmin = x0
    xmax = x0 + width * px
    ymax = y0
    ymin = y0 + height * py
    return {
        "xmin": float(min(xmin, xmax)),
        "ymin": float(min(ymin, ymax)),
        "xmax": float(max(xmin, xmax)),
        "ymax": float(max(ymin, ymax)),
    }


def robust_scale(x):
    x = x.astype("float32")
    m = np.isfinite(x) & (x != 0)
    if m.sum() == 0:
        return np.zeros_like(x, dtype="float32")
    p2, p98 = np.percentile(x[m], [2, 98])
    return np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1)


def make_thumbnail_rgb(bands: dict[str, np.ndarray], size: int = 256) -> Image.Image:
    # Approx RGB from PhiSat-2:
    # R = 665 nm B03, G = 560 nm B02, B = 490 nm B01
    r = robust_scale(bands["B03"])
    g = robust_scale(bands["B02"])
    b = robust_scale(bands["B01"])
    rgb = np.stack([r, g, b], axis=-1)
    rgb8 = (255 * rgb).clip(0, 255).astype("uint8")
    img = Image.fromarray(rgb8, mode="RGB")
    return img.resize((size, size), Image.BILINEAR)


def image_to_jpeg_bytes(img: Image.Image, quality=90) -> bytes:
    bio = BytesIO()
    img.save(bio, format="JPEG", quality=quality)
    return bio.getvalue()


def array_to_tiff_bytes(arr, crs, transform, nodata=0, dtype="uint16") -> bytes:
    arr = np.asarray(arr)

    if dtype == "uint16":
        out = np.nan_to_num(arr, nan=0.0)
        out = np.clip(out, 0, np.iinfo(np.uint16).max).astype("uint16")
    elif dtype == "uint8":
        out = np.nan_to_num(arr, nan=0.0)
        out = np.clip(out, 0, 255).astype("uint8")
    else:
        out = arr.astype(dtype)

    profile = {
        "driver": "GTiff",
        "height": out.shape[0],
        "width": out.shape[1],
        "count": 1,
        "dtype": out.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(out, 1)
        return memfile.read()


def warp_band_to_cell(src_arr, src_transform, src_crs, dst_transform, dst_crs):
    dst = np.zeros((CELL_SIZE, CELL_SIZE), dtype="float32")
    reproject(
        source=src_arr.astype("float32"),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.bilinear,
    )
    return dst


def pick_rows():
    report = pd.read_csv(BATCH_REPORT)

    # Prefer the already visually validated / comparable rows.
    report = report.sort_values(["phi_valid", "s2_valid"], ascending=False)

    rows = []
    seen_cells = set()

    for _, r in report.iterrows():
        cell_id = r["cell_id"]
        if cell_id in seen_cells:
            continue

        if float(r["phi_valid"]) < 0.999:
            continue

        raw_path = Path(r["raw_path"])
        if not raw_path.exists():
            continue

        rows.append(r)
        seen_cells.add(cell_id)

        if len(rows) >= N_SAMPLES:
            break

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)
    (OUTROOT / "index").mkdir(exist_ok=True)
    (OUTROOT / "images").mkdir(exist_ok=True)
    (OUTROOT / "metadata").mkdir(exist_ok=True)

    fits = pd.read_csv(FITS)
    mt_index = pd.read_parquet(MAJORTOM_INDEX)
    rows = pick_rows()

    print("Selected rows:")
    print(rows[["product_id", "cell_id", "country", "state", "district", "phi_valid"]].to_string(index=False))

    image_rows = []
    index_rows = []
    manifest_rows = []

    for i, r in rows.iterrows():
        pair = r["pair"]
        product_id = int(r["product_id"])
        cell_id = r["cell_id"]
        raw_path = Path(r["raw_path"])

        print(f"\nProcessing {sample_id(cell_id, product_id)}")
        print("raw:", raw_path)

        fit = fits[fits["pair"] == pair].iloc[0]
        cell = mt_index[mt_index["id"] == cell_id].iloc[0]

        gt = clean_gt(cell["majortom:geotransform"])
        x0, px, _, y0, _, py = gt

        dst_transform = Affine(px, 0, x0, 0, py, y0)
        dst_crs = CRS.from_string(cell["majortom:crs"])
        bbox = bbox_from_gt(gt)

        base_transform = Affine(
            float(fit["lon_a"]), float(fit["lon_b"]), float(fit["lon_c"]),
            float(fit["lat_d"]), float(fit["lat_e"]), float(fit["lat_f"]),
        )

        warped_bands = {}

        with rasterio.open(raw_path) as src:
            raw_w = src.width

            # Correct raw PhiSat-2 horizontal flip.
            src_transform = base_transform * Affine.translation(raw_w - 1, 0) * Affine.scale(-1, 1)

            for band_name, spec in PHISAT_BANDS.items():
                raw_idx = spec["raw_index_1based"]
                arr = src.read(raw_idx).astype("float32")
                warped = warp_band_to_cell(
                    arr,
                    src_transform=src_transform,
                    src_crs="EPSG:4326",
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                )
                warped_bands[band_name] = warped

        valid = np.zeros((CELL_SIZE, CELL_SIZE), dtype="uint8")
        ref_band = warped_bands["B03"]
        valid[(np.isfinite(ref_band)) & (ref_band != 0)] = 1
        valid_fraction = float(valid.mean())

        thumb = make_thumbnail_rgb(warped_bands)
        thumb_bytes = image_to_jpeg_bytes(thumb)

        img_row = {
            "product_id": f"PHISAT2_L1C_{product_short(product_id)}",
            "grid_cell": cell_short(cell_id),
            "product_datetime": pair_datetime_iso(pair),
            "thumbnail": thumb_bytes,
            "valid_mask": array_to_tiff_bytes(valid, dst_crs, dst_transform, nodata=0, dtype="uint8"),
        }

        for band_name, arr in warped_bands.items():
            img_row[band_name] = array_to_tiff_bytes(arr, dst_crs, dst_transform, nodata=0, dtype="uint16")

        image_rows.append(img_row)

        idx_row = cell.to_dict()

        idx_row["sample_id"] = sample_id(cell_id, product_id)
        idx_row["grid_cell"] = cell_id
        idx_row["product_id"] = product_id
        idx_row["processing_level"] = "L1C"
        idx_row["date"] = img_row["product_datetime"]

        idx_row["crs"] = str(dst_crs)
        idx_row["ul_x"] = float(x0)
        idx_row["ul_y"] = float(y0)
        idx_row["width"] = CELL_SIZE
        idx_row["height"] = CELL_SIZE
        idx_row["resolution_m"] = TARGET_RES_M
        idx_row["geotransform"] = gt
        idx_row["bbox"] = json.dumps(bbox)

        idx_row["phisat2:valid_fraction"] = valid_fraction
        idx_row["phisat2:source_path"] = str(raw_path)
        idx_row["phisat2:pair"] = pair
        idx_row["phisat2:alignment_source"] = "lorenzo_lightglue_georeferenced_patches"
        idx_row["phisat2:geometric_model"] = "full_scene_affine_from_patch_gcps"
        idx_row["phisat2:orientation_correction"] = "horizontal_flip"
        idx_row["phisat2:output_grid"] = "official_major_tom_cell_grid"
        idx_row["phisat2:output_width"] = CELL_SIZE
        idx_row["phisat2:output_height"] = CELL_SIZE
        idx_row["phisat2:output_resolution_m"] = TARGET_RES_M
        idx_row["phisat2:bands"] = json.dumps(PHISAT_BANDS)
        idx_row["phisat2:note"] = "All 8 PhiSat-2 raw bands warped to the official MajorTOM cell grid."

        # keep optional fields from batch report
        for col in ["country", "state", "district", "coverage", "s2_path", "out_png", "out_phi_tif"]:
            if col in r:
                idx_row[f"phisat2:{col}"] = r[col]

        index_rows.append(idx_row)

        manifest_rows.append({
            "sample_id": sample_id(cell_id, product_id),
            "product_id": f"PHISAT2_L1C_{product_short(product_id)}",
            "grid_cell": cell_short(cell_id),
            "major_tom_id": cell_id,
            "product_datetime": str(img_row["product_datetime"]),
            "crs": str(dst_crs),
            "width": CELL_SIZE,
            "height": CELL_SIZE,
            "resolution_m": TARGET_RES_M,
            "valid_fraction": valid_fraction,
            "source_path": str(raw_path),
            "pair": pair,
        })

    df_images = pd.DataFrame(image_rows)
    df_index = pd.DataFrame(index_rows)
    df_manifest = pd.DataFrame(manifest_rows)

    images_path = OUTROOT / "images" / "part_00001.parquet"
    index_path = OUTROOT / "index" / "global.parquet"
    manifest_path = OUTROOT / "metadata" / "manifest.csv"
    summary_path = OUTROOT / "metadata" / "dataset_summary.json"

    df_images.to_parquet(images_path, index=False)
    df_index.to_parquet(index_path, index=False)
    df_manifest.to_csv(manifest_path, index=False)

    summary = {
        "dataset_name": "PhiSat2-L1C-MajorTOM-Aligned-Demo",
        "n_samples": len(df_images),
        "format": {
            "index": "index/global.parquet",
            "images": "images/part_00001.parquet",
            "manifest": "metadata/manifest.csv",
        },
        "image_columns": list(df_images.columns),
        "band_mapping": PHISAT_BANDS,
        "geometry": {
            "target_grid": "official MajorTOM 10m cell grid",
            "width": CELL_SIZE,
            "height": CELL_SIZE,
            "resolution_m": TARGET_RES_M,
            "correction": "Lorenzo LightGlue patch-derived affine + horizontal flip",
        },
        "notes": [
            "Band columns are GeoTIFF bytes, following the Core-S2L1C parquet style.",
            "grid_cell in images parquet omits the MT10_ prefix, following Core-S2L1C convention.",
            "index/global.parquet keeps the official MajorTOM row and adds phisat2:* provenance fields.",
        ],
    }

    summary_path.write_text(json.dumps(summary, indent=2))

    readme = """# PhiSat-2 L1C MajorTOM-aligned demo

This is a small MajorTOM-like demo subset for a PhiSat-2 L1C aligned layer.

Each sample is generated by:
1. fitting a full-scene affine georeference from Lorenzo LightGlue-georeferenced patches,
2. applying this affine to the full raw PhiSat-2 acquisition,
3. correcting the known horizontal flip of the raw PhiSat-2 image,
4. warping all 8 PhiSat-2 bands to the official MajorTOM 10 m cell grid.

Files:
- `index/global.parquet`: official MajorTOM index rows enriched with PhiSat-2 provenance fields.
- `images/part_00001.parquet`: image data in parquet form, with bands stored as GeoTIFF bytes.
- `metadata/manifest.csv`: compact manifest.
- `metadata/dataset_summary.json`: schema and band mapping summary.

Band mapping:
- B00: 625 nm
- B01: 490 nm
- B02: 560 nm
- B03: 665 nm
- B04: 705 nm
- B05: 740 nm
- B06: 783 nm
- B07: 842 nm
"""
    (OUTROOT / "README.md").write_text(readme)

    print("\nSaved:")
    print(index_path)
    print(images_path)
    print(manifest_path)
    print(summary_path)
    print(OUTROOT / "README.md")

    print("\nImage columns:")
    print(df_images.columns.tolist())

    print("\nIndex columns sample:")
    print(df_index.columns.tolist()[:80])


if __name__ == "__main__":
    main()
