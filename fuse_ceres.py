"""Apply COV-adaptive CERES-guided fusion to MODELf2 DIR/DIF GeoTIFFs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage


# =============================================================================
# 1. USER SETTINGS: replace only these paths for a local run
# =============================================================================
MODELF2_DIR = Path("./MODELf2_DIRDIF")
CERES_DIR = Path("path/to/CERES")
OUTPUT_DIR = Path("./FUSED_DIRDIF")

IDW_WINDOW_SIZE = 41
COV_WINDOW_SIZE = 41
MODELF2_SCALE = 0.025
OUTPUT_SCALE = 0.025
OUTPUT_NODATA = 65535
OVERWRITE = False

DIR_BINS = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100])
DIR_WEIGHTS = np.array([1.0, 1.0, 0.9, 0.8, 0.8, 0.6, 0.5, 0.4,
                        0.4, 0.4, 0.3, 0.3, 0.1, 0.0])
DIF_BINS = np.array([5, 10, 15, 20, 25, 30, 35, 40, 65, 100])
DIF_WEIGHTS = np.array([1.0, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1, 0.0])
LUT_OVERFLOW_WEIGHT = 0.0

TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(\d{8})_(\d{4})(?!\d)")


# =============================================================================
# 2. Fusion algorithm
# =============================================================================
def create_idw_kernel(size: int) -> np.ndarray:
    if size % 2 == 0:
        size += 1
    radius = size // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    distance = np.sqrt(x**2 + y**2)
    with np.errstate(divide="ignore"):
        kernel = 1.0 / distance
    kernel[radius, radius] = 0.0
    kernel[radius, radius] = np.max(kernel) * 1.5
    return kernel.astype(np.float32)


def idw_smooth(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values).astype(np.float32)
    filled = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    weighted_sum = ndimage.convolve(filled, kernel, mode="reflect")
    weight_sum = ndimage.convolve(valid, kernel, mode="reflect")
    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = weighted_sum / weight_sum
    smoothed[weight_sum == 0] = np.nan
    return smoothed


def local_cov(values: np.ndarray, size: int) -> np.ndarray:
    valid = np.isfinite(values).astype(np.float32)
    filled = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    window_pixels = size**2

    local_sum = ndimage.uniform_filter(filled, size=size, mode="reflect") * window_pixels
    local_count = ndimage.uniform_filter(valid, size=size, mode="reflect") * window_pixels
    local_square_sum = (
        ndimage.uniform_filter(filled**2, size=size, mode="reflect") * window_pixels
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        local_mean = local_sum / local_count
        local_variance = local_square_sum / local_count - local_mean**2
    local_variance = np.maximum(local_variance, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = np.sqrt(local_variance) / local_mean * 100.0
    cov[local_mean == 0] = 0.0
    cov[~np.isfinite(cov)] = 0.0
    return cov.astype(np.float32)


def lut_weight(cov: np.ndarray, bins: np.ndarray, weights: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(bins, cov, side="right")
    complete_weights = np.append(weights, LUT_OVERFLOW_WEIGHT)
    indices = np.clip(indices, 0, len(complete_weights) - 1)
    return complete_weights[indices].astype(np.float32)


def fuse_component(
    modelf2: np.ndarray,
    ceres: np.ndarray,
    kernel: np.ndarray,
    bins: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    # Large-scale background difference between CERES and MODELf2.
    bias = idw_smooth(ceres, kernel) - idw_smooth(modelf2, kernel)
    bias[~np.isfinite(bias)] = 0.0

    # Retain more MODELf2 information under larger local heterogeneity.
    weight_map = lut_weight(local_cov(modelf2, COV_WINDOW_SIZE), bins, weights)
    fused = modelf2 + weight_map * bias
    fused[fused < 0] = 0.0
    return fused.astype(np.float32)


# =============================================================================
# 3. CERES matching, raster alignment, and output
# =============================================================================
def extract_timestamp(path: Path) -> str | None:
    match = TIMESTAMP_PATTERN.search(path.name)
    return match.group(0) if match else None


def find_ceres_files(
    satellite_name: str,
    timestamp: str,
) -> tuple[Path | None, Path | None]:
    date_string = timestamp[:8]
    dir_matches = sorted(
        (CERES_DIR / "DIR_all_sp" / satellite_name / date_string).glob(
            f"*{timestamp}*.tif"
        )
    )
    dif_matches = sorted(
        (CERES_DIR / "DIF_all_sp" / satellite_name / date_string).glob(
            f"*{timestamp}*.tif"
        )
    )
    return (
        dir_matches[0] if dir_matches else None,
        dif_matches[0] if dif_matches else None,
    )


def resample_ceres(path: Path, shape, transform, crs) -> np.ndarray:
    output = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=output,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    return output


def pack_uint16(values: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, OUTPUT_NODATA, dtype=np.uint16)
    valid = np.isfinite(values)
    if np.any(valid):
        packed = np.clip(
            np.round(values[valid] / OUTPUT_SCALE),
            0,
            OUTPUT_NODATA - 1,
        )
        output[valid] = packed.astype(np.uint16)
    return output


def fuse_scene(modelf2_path: Path, kernel: np.ndarray) -> str:
    timestamp = extract_timestamp(modelf2_path)
    if timestamp is None:
        return f"[SKIP] no timestamp: {modelf2_path.name}"
    satellite_name = modelf2_path.name.split("_", maxsplit=1)[0]
    output_path = OUTPUT_DIR / satellite_name / timestamp[:8] / (
        f"{modelf2_path.stem}_FUSED.tif"
    )
    if output_path.exists() and not OVERWRITE:
        return f"[SKIP] {output_path.name} already exists"

    ceres_dir_path, ceres_dif_path = find_ceres_files(satellite_name, timestamp)
    if ceres_dir_path is None or ceres_dif_path is None:
        return f"[SKIP] missing CERES: {satellite_name} {timestamp}"

    with rasterio.open(modelf2_path) as source:
        if source.count < 2:
            return f"[SKIP] MODELf2 file has fewer than two bands: {modelf2_path.name}"
        metadata = source.meta.copy()
        shape = (source.height, source.width)
        transform, crs, nodata = source.transform, source.crs, source.nodata
        dir_raw, dif_raw = source.read(1), source.read(2)

    modelf2_dir = dir_raw.astype(np.float32) * MODELF2_SCALE
    modelf2_dif = dif_raw.astype(np.float32) * MODELF2_SCALE
    if nodata is not None:
        invalid = (dir_raw == nodata) | (dif_raw == nodata)
        modelf2_dir[invalid] = np.nan
        modelf2_dif[invalid] = np.nan

    ceres_dir = resample_ceres(ceres_dir_path, shape, transform, crs)
    ceres_dif = resample_ceres(ceres_dif_path, shape, transform, crs)
    fused_dir = fuse_component(modelf2_dir, ceres_dir, kernel, DIR_BINS, DIR_WEIGHTS)
    fused_dif = fuse_component(modelf2_dif, ceres_dif, kernel, DIF_BINS, DIF_WEIGHTS)

    common_invalid = ~np.isfinite(fused_dir) | ~np.isfinite(fused_dif)
    fused_dir[common_invalid] = np.nan
    fused_dif[common_invalid] = np.nan

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.update(
        count=2,
        dtype="uint16",
        nodata=OUTPUT_NODATA,
        compress="lzw",
        predictor=2,
    )
    with rasterio.open(output_path, "w", **metadata) as destination:
        destination.write(pack_uint16(fused_dir), 1)
        destination.write(pack_uint16(fused_dif), 2)
        destination.scales = (OUTPUT_SCALE, OUTPUT_SCALE)
        destination.offsets = (0.0, 0.0)
        destination.set_band_description(1, "DIR")
        destination.set_band_description(2, "DIF")
    return f"[OK] {output_path}"


def main() -> None:
    kernel = create_idw_kernel(IDW_WINDOW_SIZE)
    modelf2_files = sorted(MODELF2_DIR.rglob("*.tif"))
    print(f"Found {len(modelf2_files)} MODELf2 scenes")
    for path in modelf2_files:
        print(fuse_scene(path, kernel))


if __name__ == "__main__":
    main()

