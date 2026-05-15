# 🌍 Automated Degraded Land Identification using Google Earth Engine

A fully automated remote sensing pipeline for identifying and mapping degraded land using **Sentinel-2**, **Landsat 9**, spectral indices, and **Random Forest classification** — no QGIS or manual shapefiles required.

---

## ✅ Features

| Feature | Description |
|---|---|
| 🛰️ Multi-sensor | Sentinel-2 SR + Landsat 9 L2 |
| 📐 Spectral Indices | NDVI, BSI, SAVI, EVI, NDWI |
| 🌡️ LST | Land Surface Temperature from Landsat 9 |
| 🤖 ML Classification | Random Forest (50 trees) |
| 🏷️ Auto Labeling | Rule-based pseudo-label generation |
| 🔄 Change Detection | NDVI & BSI change from baseline (2018–19) |
| 📊 Accuracy Assessment | Confusion matrix + Kappa coefficient |
| 📤 GeoTIFF Export | Exports to Google Drive |
| 🗺️ Interactive Map | `geemap` visualization |
| 📈 Area Statistics | Per-class area in km² with bar chart |

---

## 🗺️ Study Area

**Location:** Rajasthan, India
**AOI:** `[73.5°E, 26.0°N]` → `[74.5°E, 27.0°N]` (1° × 1° bounding box)

| Period | Date Range |
|---|---|
| **Current** | Nov 2023 – Feb 2024 |
| **Baseline** | Nov 2018 – Feb 2019 |

---

## 📦 Installation

```bash
pip install earthengine-api geemap matplotlib numpy pandas
```

### First-time Authentication

```bash
earthengine authenticate
```

Then update the project ID in the script:

```python
ee.Initialize(project='your-gee-project-id')
```

---

## 🚀 Usage

```bash
python degraded_land_gee.py
```

The script runs end-to-end automatically — no manual inputs needed.

---

## 🔬 Methodology

### 1. Cloud Masking
- **Sentinel-2:** QA60 band — bits 10 & 11 (opaque + cirrus clouds)
- **Landsat 9:** QA_PIXEL band — bits 3 & 4 (cloud shadow + cloud)

### 2. Spectral Indices

| Index | Formula | Use |
|---|---|---|
| **NDVI** | (B8 − B4) / (B8 + B4) | Vegetation health |
| **NDWI** | (B3 − B8) / (B3 + B8) | Water content |
| **SAVI** | 1.5 × (B8 − B4) / (B8 + B4 + 0.5) | Soil-adjusted vegetation |
| **EVI** | 2.5 × (B8 − B4) / (B8 + 6×B4 − 7.5×B2 + 1) | Enhanced vegetation |
| **BSI** | (B11 + B4 − B8 − B2) / (B11 + B4 + B8 + B2) | Bare soil |

### 3. Degradation Classes

| Class | Label | Criteria |
|---|---|---|
| 0 | None | Healthy vegetation |
| 1 | Mild | NDVI < 0.3 & BSI > 0.00 |
| 2 | Moderate | NDVI < 0.2 & BSI > 0.05 |
| 3 | Severe | NDVI < 0.1 & BSI > 0.10 |

### 4. Random Forest Classification
- **Training:** 100 stratified pseudo-labeled samples per class (seed=42)
- **Validation:** 50 stratified samples (seed=7)
- **Features:** NDVI, NDWI, SAVI, EVI, BSI, B2, B3, B4, B8, B11
- **Trees:** 50

### 5. Change Detection
- NDVI change = Current NDVI − Baseline NDVI
- BSI change = Current BSI − Baseline BSI

---

## 📤 Outputs

All exports go to Google Drive under the `RemoteSensing_Output/` folder.

| File | Type | Description |
|---|---|---|
| `NDVI_Visual.tif` | GeoTIFF (RGB) | NDVI colored map |
| `BSI_Visual.tif` | GeoTIFF (RGB) | BSI colored map |
| `RF_Degraded_Land_Map.tif` | GeoTIFF (RGB) | RF classification colored map |
| `NDVI_Change_Visual.tif` | GeoTIFF (RGB) | NDVI change colored map |
| `NDVI_RAW.tif` | GeoTIFF (Float) | Raw NDVI values |
| `BSI_RAW.tif` | GeoTIFF (Float) | Raw BSI values |
| `RF_Classification_RAW.tif` | GeoTIFF (Float) | Raw class labels (0–3) |
| `LST_RAW.tif` | GeoTIFF (Float) | Land Surface Temp (°C) |
| `degradation_area_statistics.png` | PNG | Bar chart of class areas |

---

## 🗺️ Map Visualization

An interactive `geemap` map is generated with four toggleable layers:

- 🟢 **NDVI** — Red → Yellow → Green
- 🔵 **BSI** — Blue → White → Red
- 🎨 **RF Classification** — Green (None) → Yellow (Mild) → Orange (Moderate) → Red (Severe)
- 🔄 **NDVI Change** — Red (loss) → White (stable) → Green (gain)

---

## 📁 Project Structure

```
degraded_land_gee.py          # Main script
degradation_area_statistics.png  # Output bar chart
README.md                        # This file
```

---

## ⚠️ Notes

- Accuracy assessment may **time out** for large AOIs — this is handled gracefully.
- Increase `numPoints` in `stratifiedSample()` for better model accuracy.
- Change the AOI coordinates to analyze any region globally.
- Ensure your GEE project has sufficient quota for exports.

---

## 📜 License

This project is open-source and free to use for research and educational purposes.

---

## 🙏 Acknowledgements

- [Google Earth Engine](https://earthengine.google.com/)
- [Copernicus / ESA — Sentinel-2](https://sentinel.esa.int/)
- [USGS — Landsat 9](https://www.usgs.gov/landsat-missions/landsat-9)
- [geemap](https://geemap.org/) by Qiusheng Wu
