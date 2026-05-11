# RemoteSensingforDegradedLand
Remote Sensing-Based Identification of Degraded Land using Sentinel-2 NDVI
"""
=============================================================================
AUTOMATED DEGRADED LAND IDENTIFICATION USING GOOGLE EARTH ENGINE
=============================================================================

FEATURES:
    ✅ Fully automatic workflow
    ✅ No QGIS required
    ✅ No manual shapefiles
    ✅ Sentinel-2 + Landsat 9
    ✅ NDVI / BSI / SAVI / EVI / NDWI
    ✅ Land Surface Temperature (LST)
    ✅ Random Forest Classification
    ✅ Automatic pseudo-label generation
    ✅ Change Detection
    ✅ Accuracy Assessment
    ✅ GeoTIFF Export
    ✅ Interactive Visualization

INSTALL:
    pip install earthengine-api geemap matplotlib numpy pandas

FIRST TIME:
    earthengine authenticate

RUN:
    python degraded_land_project.py
=============================================================================
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

import ee
import geemap
import matplotlib.pyplot as plt

# =============================================================================
# 1. AUTHENTICATION & INITIALIZATION
# =============================================================================

ee.Authenticate()

ee.Initialize(project='ndvimaps')

print("✅ Google Earth Engine initialized.")

# =============================================================================
# 2. AREA OF INTEREST (SMALLER AOI FOR FAST PROCESSING)
# =============================================================================

# Smaller Rajasthan AOI
aoi = ee.Geometry.Rectangle([73.5, 26.0, 74.5, 27.0])

# Current period
start_date = '2023-11-01'
end_date   = '2024-02-28'

# Baseline period
baseline_start = '2018-11-01'
baseline_end   = '2019-02-28'

print("✅ AOI and dates configured.")

# =============================================================================
# 3. CLOUD MASKING
# =============================================================================

def mask_s2_clouds(image):

    qa = image.select('QA60')

    mask = (
        qa.bitwiseAnd(1 << 10).eq(0)
        .And(qa.bitwiseAnd(1 << 11).eq(0))
    )

    return image.updateMask(mask).divide(10000)


def mask_landsat_clouds(image):

    qa = image.select('QA_PIXEL')

    mask = (
        qa.bitwiseAnd(1 << 3).eq(0)
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )

    return (
        image.updateMask(mask)
        .multiply(0.0000275)
        .add(-0.2)
    )

print("✅ Cloud masking functions ready.")

# =============================================================================
# 4. LOAD SATELLITE DATA
# =============================================================================

# Sentinel-2
s2 = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(mask_s2_clouds)
    .median()
    .clip(aoi)
)

# Landsat 9
ls9 = (
    ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(start_date, end_date)
    .map(mask_landsat_clouds)
    .median()
    .clip(aoi)
)

# Baseline Sentinel-2
s2_baseline = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(baseline_start, baseline_end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(mask_s2_clouds)
    .median()
    .clip(aoi)
)

print("✅ Satellite imagery loaded.")

# =============================================================================
# 5. COMPUTE SPECTRAL INDICES
# =============================================================================

def compute_indices(img):

    B2  = img.select('B2')
    B3  = img.select('B3')
    B4  = img.select('B4')
    B8  = img.select('B8')
    B11 = img.select('B11')

    # NDVI
    ndvi = (
        img.normalizedDifference(['B8', 'B4'])
        .rename('NDVI')
    )

    # NDWI
    ndwi = (
        img.normalizedDifference(['B3', 'B8'])
        .rename('NDWI')
    )

    # SAVI
    savi = (
        B8.subtract(B4)
        .multiply(1.5)
        .divide(B8.add(B4).add(0.5))
        .rename('SAVI')
    )

    # EVI
    evi = (
        B8.subtract(B4)
        .multiply(2.5)
        .divide(
            B8.add(B4.multiply(6))
            .subtract(B2.multiply(7.5))
            .add(1)
        )
        .rename('EVI')
    )

    # BSI
    bsi = (
        (B11.add(B4))
        .subtract(B8.add(B2))
        .divide((B11.add(B4)).add(B8).add(B2))
        .rename('BSI')
    )

    return img.addBands([
        ndvi,
        ndwi,
        savi,
        evi,
        bsi
    ])

# Current indices
s2_indices = compute_indices(s2)

# Baseline indices
s2_baseline_indices = compute_indices(s2_baseline)

print("✅ Spectral indices computed.")

# =============================================================================
# 6. LAND SURFACE TEMPERATURE
# =============================================================================

lst = (
    ls9.select('ST_B10')
    .multiply(0.00341802)
    .add(149.0)
    .subtract(273.15)
    .rename('LST')
)

print("✅ Land Surface Temperature computed.")

# =============================================================================
# 7. RULE-BASED DEGRADATION MAP
# =============================================================================

ndvi = s2_indices.select('NDVI')
bsi  = s2_indices.select('BSI')

threshold_map = (
    ee.Image(0)
    .where(ndvi.lt(0.3).And(bsi.gt(0.00)), 1)
    .where(ndvi.lt(0.2).And(bsi.gt(0.05)), 2)
    .where(ndvi.lt(0.1).And(bsi.gt(0.10)), 3)
    .rename('degradation_class')
    .clip(aoi)
)

print("✅ Threshold degradation map created.")

# =============================================================================
# 8. AUTO TRAINING & VALIDATION DATA
# =============================================================================

pseudo_labels = (
    ee.Image(0)
    .where(ndvi.lt(0.3).And(bsi.gt(0.00)), 1)
    .where(ndvi.lt(0.2).And(bsi.gt(0.05)), 2)
    .where(ndvi.lt(0.1).And(bsi.gt(0.10)), 3)
    .rename('class')
)

# Training samples
training_pts = pseudo_labels.stratifiedSample(
    numPoints=100,
    classBand='class',
    region=aoi,
    scale=30,
    seed=42,
    geometries=True
)

# Validation samples
validation_pts = pseudo_labels.stratifiedSample(
    numPoints=50,
    classBand='class',
    region=aoi,
    scale=30,
    seed=7,
    geometries=True
)

print("✅ Automatic samples generated.")

# =============================================================================
# 9. FEATURE STACK
# =============================================================================

feature_stack = (
    s2_indices.select([
        'NDVI',
        'NDWI',
        'SAVI',
        'EVI',
        'BSI',
        'B2',
        'B3',
        'B4',
        'B8',
        'B11'
    ])
)

print("✅ Feature stack prepared.")

# =============================================================================
# 10. RANDOM FOREST CLASSIFICATION
# =============================================================================

training = feature_stack.sampleRegions(
    collection=training_pts,
    properties=['class'],
    scale=30
)

rf_classifier = (
    ee.Classifier.smileRandomForest(
        numberOfTrees=50,
        seed=42
    )
    .train(
        features=training,
        classProperty='class',
        inputProperties=feature_stack.bandNames()
    )
)

classified_map = (
    feature_stack
    .classify(rf_classifier)
    .rename('RF_class')
)

print("✅ Random Forest classification completed.")

# =============================================================================
# 11. CHANGE DETECTION
# =============================================================================

ndvi_change = (
    s2_indices.select('NDVI')
    .subtract(s2_baseline_indices.select('NDVI'))
    .rename('NDVI_change')
)

bsi_change = (
    s2_indices.select('BSI')
    .subtract(s2_baseline_indices.select('BSI'))
    .rename('BSI_change')
)

print("✅ Change detection completed.")

# =============================================================================
# 12. ACCURACY ASSESSMENT
# =============================================================================

validated = classified_map.sampleRegions(
    collection=validation_pts,
    properties=['class'],
    scale=30
)

confusion_matrix_gee = validated.errorMatrix(
    'class',
    'RF_class'
)

print("\n📊 ACCURACY ASSESSMENT")
print("=" * 40)

try:

    print(
        "Overall Accuracy:",
        confusion_matrix_gee.accuracy().getInfo()
    )

    print(
        "Kappa Coefficient:",
        confusion_matrix_gee.kappa().getInfo()
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix_gee.getInfo())

except Exception as e:

    print("⚠️ Accuracy computation timeout.")
    print("Error:", e)

# =============================================================================
# 13. EXPORT FUNCTION
# =============================================================================

def export_image(
    image,
    description,
    folder='RemoteSensing_Output',
    scale=30,
    crs='EPSG:4326'
):

    task = ee.batch.Export.image.toDrive(
        image=image.toFloat(),
        description=description,
        folder=folder,
        region=aoi,
        scale=scale,
        crs=crs,
        fileFormat='GeoTIFF',
        maxPixels=1e13
    )

    task.start()

    print(f"▶️ Export started: {description}")

# =============================================================================
# 14. VISUALIZATION EXPORTS (COLORFUL MAPS)
# =============================================================================

# NDVI visualization
ndvi_vis = s2_indices.select('NDVI').visualize(
    min=-0.1,
    max=0.8,
    palette=['red', 'yellow', 'green']
)

# BSI visualization
bsi_vis = s2_indices.select('BSI').visualize(
    min=-0.5,
    max=0.5,
    palette=['blue', 'white', 'red']
)

# RF classification visualization
rf_vis = classified_map.visualize(
    min=0,
    max=3,
    palette=[
        '2ECC71',
        'F7DC6F',
        'E67E22',
        'C0392B'
    ]
)

# Change detection visualization
ndvi_change_vis = ndvi_change.visualize(
    min=-0.5,
    max=0.5,
    palette=['red', 'white', 'green']
)

print("✅ Visualization layers created.")

# =============================================================================
# 15. EXPORT MAPS
# =============================================================================

print("\n📤 Exporting maps to Google Drive...")

# Color exports
export_image(ndvi_vis, 'NDVI_Visual')
export_image(bsi_vis, 'BSI_Visual')
export_image(rf_vis, 'RF_Degraded_Land_Map')
export_image(ndvi_change_vis, 'NDVI_Change_Visual')

# Scientific raster exports
export_image(
    s2_indices.select('NDVI'),
    'NDVI_RAW'
)

export_image(
    s2_indices.select('BSI'),
    'BSI_RAW'
)

export_image(
    classified_map,
    'RF_Classification_RAW'
)

export_image(
    lst,
    'LST_RAW'
)

print("✅ Export tasks submitted.")

# =============================================================================
# 16. INTERACTIVE MAP
# =============================================================================

Map = geemap.Map()

Map.centerObject(aoi, 9)

Map.addLayer(
    ndvi_vis,
    {},
    'NDVI'
)

Map.addLayer(
    bsi_vis,
    {},
    'BSI'
)

Map.addLayer(
    rf_vis,
    {},
    'RF Classification'
)

Map.addLayer(
    ndvi_change_vis,
    {},
    'NDVI Change'
)

Map.addLayerControl()

print("✅ Interactive map ready.")

Map

# =============================================================================
# 17. AREA STATISTICS
# =============================================================================

pixel_area = (
    classified_map.eq([0, 1, 2, 3])
    .multiply(ee.Image.pixelArea())
    .divide(1e6)
)

class_areas = pixel_area.reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=aoi,
    scale=30,
    maxPixels=1e13
).getInfo()

labels = [
    'None',
    'Mild',
    'Moderate',
    'Severe'
]

print("\n📊 AREA STATISTICS")
print("=" * 40)

total_area = sum(class_areas.values())

for key, label in zip(class_areas.keys(), labels):

    area_km2 = class_areas[key]

    pct = (
        (area_km2 / total_area) * 100
        if total_area > 0 else 0
    )

    print(
        f"{label:10s}: "
        f"{area_km2:.2f} km² "
        f"({pct:.1f}%)"
    )

# =============================================================================
# 18. BAR CHART
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    labels,
    list(class_areas.values())
)

ax.set_xlabel('Degradation Class')
ax.set_ylabel('Area (km²)')
ax.set_title('Degraded Land Area Statistics')

for bar, val in zip(bars, class_areas.values()):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{val:.1f}',
        ha='center',
        va='bottom'
    )

plt.tight_layout()

plt.savefig(
    'degradation_area_statistics.png',
    dpi=150
)

plt.show()

print("✅ Chart saved.")

# =============================================================================
# END
# =============================================================================

print("\n🎉 DEGRADED LAND ANALYSIS COMPLETED.")
