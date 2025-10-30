# Vegetation Canopy Cover Calculator in Python 3.13 (vCcCpy)

[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for calculating vegetation and canopy cover ratios from high-resolution raster (≤10m) data for polygon features. Supports processing of large polygons through memory-efficient chunkwise processing and intelligent splitting.

## Features

- **Memory Efficient**: Handles large datasets through chunk processing and optimized memory usage
- **Intelligent Splitting**: Automatically splits large polygons to prevent memory issues
- **Multiple Outputs**: Calculate both vegetation cover ratios (VCr) and areas (VCa)
- **Flexible Input**: Supports various raster and vector formats
- **Progress Tracking**: Real-time progress monitoring with tqdm
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Installation

### Option 1: Install directly from GitHub
```bash
## Version 1:
pip install git+https://github.com/DijoG/vCcCpy.git
```

### Option 2: Clone and install in development mode
```bash
git clone https://github.com/DijoG/vCcCpy.git
cd vCcCpy
pip install -e .
```

### Dependencies

- GDAL (≥3.0)
- NumPy (≥1.20)
- tqdm (≥4.60)
- pandas (≥1.3)
- geopandas (≥0.14.0)
- rasterio (≥1.3.8)
- rioxarray (≥0.15.0)
- shapely (≥2.0.2)
- pyproj (≥3.6.1)
- dask (≥2023.10.0)
- dask-geopandas (≥0.3.0)

## Usage Example

### Basic Usage - Chunkwise Processing (Recommended)

```python
from vCcCpy.core import get_VEGETATION

# Process without pre-splitting (most efficient)
results = get_VEGETATION(
    raster_path="path/to/binarized_raster.tif",
    vector_path="path/to/polygons.gpkg",
    output_path="path/to/output.gpkg",
    split_polygons=False  # Use chunkwise processing
)
```
### Advanced Usage - With Pre-splitting

```python
import geopandas as gpd
from vCcCpy.core import get_VEGETATION, explode_pid, aggregate_by_field
from vCcCpy.splitter import split_large_polygons, analyze_polygon_sizes

# Load and prepare data
GRP = explode_pid("path/to/vector.geojson", field_to_string='MCAT')

# Filter specific categories for testing
GRP_test = GRP[GRP['MCAT'].isin(["Large_Parks", "Wadis"])].copy()

# Step 1: Define splitting strategies based on analysis
strategies = {
    "Wadis": {"threshold": 5000000, "n_areas": 30},
    "Large_Parks": {"threshold": 200000, "n_areas": 20}
}

# Step 2: Pre-split large polygons
GRP_pre_split = split_large_polygons(
    gdf=GRP_test, 
    category_field="MCAT", 
    splitting_strategies=strategies,
    id_field="pid"
)

# Step 3: Process split polygons
result = get_VEGETATION(
    polygons=GRP_pre_split,
    veg_raster="path/to/binarized_raster.tif",
    output_path="path/to/split_output.gpkg",
    id_field="pid",
    by_row=True,
    return_result=True
)

# Step 4: Aggregate processed polygons
final_results = aggregate_by_field(
    input_path="path/to/split_output.gpkg",
    output_path="path/to/final_aggregated.gpkg",
    field_name="pid"  # or 'MCAT', a field containing original polygon identifiers
)
```
### Calculate Vegetation Cover Ratios and Areas

```python
from vCcCpy.core import get_VCratio, get_VCarea

# Calculate vegetation cover ratio
ratio_results = get_VCratio(
    raster_stack_path="path/to/stacked_raster.tif",
    vector_path="path/to/polygons.gpkg",
    output_path="path/to/ratio_output.gpkg"
)

# Calculate vegetation cover area
area_results = get_VCarea(
    raster_stack_path="path/to/stacked_raster.tif", 
    vector_path="path/to/polygons.gpkg",
    output_path="path/to/area_output.gpkg"
)
```
## Output

The package generates output files in various geospatial formats (GeoPackage, GeoJSON, Shapefile) containing:

- Original polygon geometries
- Calculated vegetation cover ratios (VCr)
- Calculated vegetation cover areas (VCa)
- Processing metadata and confidence metrics

## Test Scripts

The package includes comprehensive test scripts in the test directory:

- `test_01aa.py`: Chunkwise processing without pre-splitting using `get_VEGETATION()`
- `test_01a.py` + `test_01b.py`: Pre-splitting with `get_VEGETATION()` and feature aggregation with `aggregate_by_field()`
- `test_02.py`: Vegetation ratio and area computation with `get_VCratio()` and `get_VCarea()`

## Important Notes

- Input raster MUST be binarized (1 = vegetation/canopy, 0 = everything else)
- Default processing: Use `get_VEGETATION()` directly for most cases
- Pre-splitting is only recommended for extremely large files → use `test_01a.py` + `test_01b.py`
- Splitting strategies: Define different thresholds for different polygon categories
- `get_VEGETATION()` was developed and tested with 0.3m resolution raster data.
- `get_VCratio()` and `get_VCarea()` were developed and tested with 10m resolution STACKED raster data

## Memory Management

For optimal performance:

- Use default chunkwise processing for most cases
- Only enable pre-splitting for extremely large datasets (>1GB)
- Ensure adequate RAM availability for processing
- Monitor progress through the built-in progress bars

Happy coding! 🌿
