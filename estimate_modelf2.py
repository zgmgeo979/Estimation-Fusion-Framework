"""Run MODELf2 inference to generate two-band DIR/DIF GeoTIFFs."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject

from mtl_embedding import MTLEmbModel


# =============================================================================
# 1. USER SETTINGS: replace only these paths for a local run
# =============================================================================
MODEL_PATH = Path("./MODELf2.pth")
SCALER_X_PATH = Path("./scaler_X_numeric.pkl")
SCALER_Y_PATH = Path("./scaler_Y.pkl")

DEM_PATH = Path("path/to/DEM_data.tif")
ALBEDO_DIR = Path("path/to/surface_albedo_tifs")
OUTPUT_DIR = Path("./MODELf2_DIRDIF")


SATELLITES = {
    "GOES-17": {
        "embedding_index": 0,
        "toa_dir": Path("path/to/GOES-17/red_band_TOA"),
        "sza_dir": Path("path/to/GOES-17/SZA"),
        "vza_path": Path("path/to/GOES-17_VZA.tif"),
    },
    "GOES-16": {
        "embedding_index": 1,
        "toa_dir": Path("path/to/GOES-16/red_band_TOA"),
        "sza_dir": Path("path/to/GOES-16/SZA"),
        "vza_path": Path("path/to/GOES-16_VZA.tif"),
    },
    "MSG-East": {
        "embedding_index": 2,
        "toa_dir": Path("path/to/MSG-East/red_band_TOA"),
        "sza_dir": Path("path/to/MSG-East/SZA"),
        "vza_path": Path("path/to/MSG-East_VZA.tif"),
    },
    "Himawari8": {
        "embedding_index": 3,
        "toa_dir": Path("path/to/Himawari8/red_band_TOA"),
        "sza_dir": Path("path/to/Himawari8/SZA"),
        "vza_path": Path("path/to/Himawari8_VZA.tif"),
    },
}

# Set to a list such as ["01", "10", "20"] to restrict processing by day.
TARGET_DAYS = None
INFERENCE_BATCH_SIZE = 262_144
TORCH_DEVICE = "cpu"
TORCH_THREADS = 1
OVERWRITE = False

TOA_MIN, TOA_MAX = 0.0, 3.0
ALBEDO_MIN, ALBEDO_MAX = 0.0, 1.0
SZA_MIN, SZA_MAX = 0.0, 85.0
VZA_MIN, VZA_MAX = 0.0, 90.0
RAD_MIN, RAD_MAX = 0.0, 1400.0

# Packed output: physical_value = stored_value * OUTPUT_SCALE.
OUTPUT_SCALE = 0.025
OUTPUT_NODATA = 65535

TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(\d{8})_(\d{4})(?!\d)")
DATE_PATTERN = re.compile(r"(?<!\d)(\d{8})(?!\d)")


# =============================================================================
# 2. File indexing and raster utilities
# =============================================================================
def extract_timestamp(path: Path) -> str | None:
    match = TIMESTAMP_PATTERN.search(path.name)
    return match.group(0) if match else None


def build_timestamp_index(root: Path) -> dict[str, Path]:
    """Index GeoTIFFs containing YYYYMMDD_HHMM in their filenames."""
    grouped: dict[str, list[Path]] = {}
    allowed_days = None if TARGET_DAYS is None else {str(day).zfill(2) for day in TARGET_DAYS}
    for path in root.rglob("*.tif"):
        timestamp = extract_timestamp(path)
        if timestamp is None:
            continue
        if allowed_days is not None and timestamp[6:8] not in allowed_days:
            continue
        grouped.setdefault(timestamp, []).append(path)
    # If duplicates occur, use the lexicographically last path deterministically.
    return {timestamp: sorted(paths)[-1] for timestamp, paths in grouped.items()}


def find_nearest_sza(sza_index: dict[str, Path], timestamp: str) -> Path | None:
    date_string, hhmm = timestamp.split("_")
    target_minutes = int(hhmm[:2]) * 60 + int(hhmm[2:])
    best_path, best_difference = None, None
    for candidate_timestamp, path in sza_index.items():
        if not candidate_timestamp.startswith(date_string + "_"):
            continue
        candidate_hhmm = candidate_timestamp[-4:]
        candidate_minutes = int(candidate_hhmm[:2]) * 60 + int(candidate_hhmm[2:])
        difference = abs(candidate_minutes - target_minutes)
        if best_difference is None or difference < best_difference:
            best_path, best_difference = path, difference
    if best_difference is not None and best_difference <= 1:
        return best_path
    return None


def nearest_albedo(albedo_paths: list[Path], timestamp: str) -> Path | None:
    target_date = datetime.strptime(timestamp[:8], "%Y%m%d")
    dated_paths = []
    for path in albedo_paths:
        match = DATE_PATTERN.search(path.name)
        if match:
            try:
                dated_paths.append((path, datetime.strptime(match.group(1), "%Y%m%d")))
            except ValueError:
                pass
    if not dated_paths:
        return None
    return min(dated_paths, key=lambda item: abs(item[1] - target_date))[0]


def read_to_grid(
    path: Path,
    *,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
) -> np.ndarray:
    """Nearest-neighbour alignment to the current TOA grid."""
    output = np.full(dst_shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as source:
        source_crs = source.crs or CRS.from_epsg(4326)
        reproject(
            source=rasterio.band(source, 1),
            destination=output,
            src_transform=source.transform,
            src_crs=source_crs,
            src_nodata=source.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
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


def load_model_and_scalers():
    """Load the MODELf2 state_dict and fitted X/Y scalers."""
    model = MTLEmbModel(numeric_input_dim=5, num_sats=4)
    try:
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)

    device = torch.device(TORCH_DEVICE)
    model.to(device)
    model.eval()
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return model, scaler_x, scaler_y, device


# =============================================================================
# 3. MODELf2 inference
# =============================================================================
def estimate_scene(
    toa_path: Path,
    timestamp: str,
    satellite_name: str,
    satellite_index: int,
    sza_index: dict[str, Path],
    vza_path: Path,
    albedo_paths: list[Path],
    model,
    scaler_x,
    scaler_y,
    device,
) -> str:
    output_path = OUTPUT_DIR / satellite_name / timestamp[:8] / (
        f"{satellite_name}_{timestamp}_DIRDIF.tif"
    )
    if output_path.exists() and not OVERWRITE:
        return f"[SKIP] {output_path.name} already exists"

    sza_path = find_nearest_sza(sza_index, timestamp)
    albedo_path = nearest_albedo(albedo_paths, timestamp)
    if sza_path is None or albedo_path is None:
        return f"[SKIP] {satellite_name} {timestamp}: missing SZA or albedo"

    with rasterio.open(toa_path) as toa_source:
        dst_height, dst_width = toa_source.height, toa_source.width
        dst_shape = (dst_height, dst_width)
        dst_transform = toa_source.transform
        dst_crs = toa_source.crs or CRS.from_epsg(4326)

    read_kwargs = {
        "dst_shape": dst_shape,
        "dst_transform": dst_transform,
        "dst_crs": dst_crs,
    }
    toa = read_to_grid(toa_path, **read_kwargs)
    sza = read_to_grid(sza_path, **read_kwargs)
    vza = read_to_grid(vza_path, **read_kwargs)
    dem = read_to_grid(DEM_PATH, **read_kwargs)
    albedo = read_to_grid(albedo_path, **read_kwargs)
    cos_sza = np.cos(np.radians(sza))
    cos_vza = np.cos(np.radians(vza))

    # Exact MODELf2 feature order used by scaler_X_numeric.pkl.
    features = np.stack([cos_sza, cos_vza, dem, albedo, toa], axis=-1).astype(np.float32)
    valid_mask = (
        np.all(np.isfinite(features), axis=-1)
        & (sza >= SZA_MIN)
        & (sza < SZA_MAX)
        & (vza >= VZA_MIN)
        & (vza <= VZA_MAX)
        & (toa >= TOA_MIN)
        & (toa <= TOA_MAX)
        & (albedo >= ALBEDO_MIN)
        & (albedo <= ALBEDO_MAX)
    )

    valid_indices = np.flatnonzero(valid_mask.ravel())
    if valid_indices.size == 0:
        return f"[SKIP] {satellite_name} {timestamp}: no valid pixels"

    valid_features = features.reshape(-1, 5)[valid_indices]
    scaled_features = np.asarray(scaler_x.transform(valid_features), dtype=np.float32)

    prediction_batches = []
    with torch.no_grad():
        for start in range(0, scaled_features.shape[0], INFERENCE_BATCH_SIZE):
            stop = min(start + INFERENCE_BATCH_SIZE, scaled_features.shape[0])
            numeric_tensor = torch.as_tensor(
                scaled_features[start:stop], dtype=torch.float32, device=device
            )
            satellite_tensor = torch.full(
                (stop - start,), satellite_index, dtype=torch.long, device=device
            )
            prediction_batches.append(
                model(numeric_tensor, satellite_tensor).cpu().numpy()
            )

    prediction_scaled = np.concatenate(prediction_batches, axis=0)
    prediction = np.asarray(scaler_y.inverse_transform(prediction_scaled), dtype=np.float32)
    prediction = np.clip(prediction, RAD_MIN, RAD_MAX)

    dir_values = np.full(dst_shape, np.nan, dtype=np.float32)
    dif_values = np.full(dst_shape, np.nan, dtype=np.float32)
    dir_values.reshape(-1)[valid_indices] = prediction[:, 0]
    dif_values.reshape(-1)[valid_indices] = prediction[:, 1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=dst_height,
        width=dst_width,
        count=2,
        dtype="uint16",
        crs=dst_crs,
        transform=dst_transform,
        nodata=OUTPUT_NODATA,
        compress="lzw",
        predictor=2,
    ) as destination:
        destination.write(pack_uint16(dir_values), 1)
        destination.write(pack_uint16(dif_values), 2)
        destination.scales = (OUTPUT_SCALE, OUTPUT_SCALE)
        destination.set_band_description(1, "DIR")
        destination.set_band_description(2, "DIF")
    return f"[OK] {output_path}"


def main() -> None:
    torch.set_num_threads(TORCH_THREADS)
    required_files = [MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH, DEM_PATH]
    required_files.extend(values["vza_path"] for values in SATELLITES.values())
    missing = [path for path in required_files if not path.is_file()]
    missing.extend(
        path
        for values in SATELLITES.values()
        for path in [values["toa_dir"], values["sza_dir"]]
        if not path.is_dir()
    )
    if not ALBEDO_DIR.is_dir():
        missing.append(ALBEDO_DIR)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Replace the placeholder paths in the USER SETTINGS section. "
            f"Missing paths:\n{formatted}"
        )

    model, scaler_x, scaler_y, device = load_model_and_scalers()
    albedo_paths = sorted(ALBEDO_DIR.rglob("*.tif"))
    if not albedo_paths:
        raise FileNotFoundError(f"No dated albedo GeoTIFFs found beneath {ALBEDO_DIR}")

    for satellite_name, settings in SATELLITES.items():
        toa_index = build_timestamp_index(settings["toa_dir"])
        sza_index = build_timestamp_index(settings["sza_dir"])
        print(f"{satellite_name}: {len(toa_index)} TOA scenes")
        for timestamp, toa_path in sorted(toa_index.items()):
            result = estimate_scene(
                toa_path,
                timestamp,
                satellite_name,
                int(settings["embedding_index"]),
                sza_index,
                settings["vza_path"],
                albedo_paths,
                model,
                scaler_x,
                scaler_y,
                device,
            )
            print(result)


if __name__ == "__main__":
    main()

