# MODELf2 inference and CERES-guided fusion

This repository contains the minimum public implementation needed to inspect
and run the radiation-estimation workflow described in the associated article.
It includes only three Python scripts:

- `mtl_embedding.py`: MODELf2 multi-task network with satellite embedding;
- `estimate_modelf2.py`: input preparation and MODELf2 DIR/DIF inference;
- `fuse_ceres.py`: COV-adaptive, CERES-guided fusion of MODELf2 DIR/DIF.

## Dependencies

Python 3.10 or later is recommended. Install:

```bash
pip install numpy scipy rasterio torch scikit-learn joblib
```

## Input data

Replace the generic `path/to/...` entries in
the **USER SETTINGS** sections of the two run scripts. MODELf2 requires:

- red-band TOA observations for each geostationary satellite;
- temporally matched SZA rasters;
- one static VZA raster for each satellite;
- a DEM raster;
- dated surface-albedo rasters.


The five MODELf2 numeric inputs, in the exact order expected by
`scaler_X_numeric.pkl`, are:

```text
[cos(SZA), cos(VZA), DEM, surface albedo, red-band TOA reflectance]
```

Satellite embedding indices are GOES-17=0, GOES-16=1, MSG-East=2, and
Himawari8=3. This mapping must match the model-training order.

The fusion script expects CERES files beneath:

```text
CERES_ROOT/
  DIR_all_sp/SATELLITE/YYYYMMDD/CERES_YYYYMMDD_HHMM_DIR_all_*.tif
  DIF_all_sp/SATELLITE/YYYYMMDD/CERES_YYYYMMDD_HHMM_DIF_all_*.tif
```

## Run

First edit the path settings, then run:

```bash
python estimate_modelf2.py
python fuse_ceres.py
```

Both stages write two-band uint16 GeoTIFFs: band 1 is DIR and band 2 is DIF.
The stored values use a scale factor of 0.025 W m-2 and nodata value 65535.

## Scope and data availability

These scripts provide the source code for MODELf2 inference and the subsequent
CERES-guided fusion procedure. The processed data and estimated radiation fields supporting the
findings of the study are available from the corresponding author upon
reasonable request. Original input datasets can be obtained from the providers
identified in the article.

