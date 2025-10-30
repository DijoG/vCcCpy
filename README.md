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

### Ottion 2: Clone and install in development mode
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

## Important Notes

- Input raster **MUST BE** binarized (1 = vegetation or canopy, 0 = everything else)
- Use **default (no pre-splitting) chunkwise processing**!
- Pre-splitting is only recommended for very large files or low computational resources 
- `get_VEGETATION()` was developed and tested with 0.3m resolution raster data.
- `get_VCratio()` and `get_VCarea()` were developed and tested with 10m resolution STACKED raster data

## Usage Example

### Basic Usage - Chunkwise Processing (Recommended)

```bash
from vCcCpy import get_VEGETATION

# Process without pre-splitting (most efficient)
results = get_VEGETATION(
    raster_path="path/to/binarized_raster.tif",
    vector_path="path/to/polygons.gpkg",
    output_path="path/to/output.gpkg",
    split_polygons=False  # Use chunkwise processing
)
```
### Advanced Usage - With Pre-splitting

```bash
from vCcCpy import get_VEGETATION, aggregate_by_field

# Step 1: Split and process large polygons
split_results = get_VEGETATION(
    raster_path="path/to/binarized_raster.tif",
    vector_path="path/to/polygons.gpkg",
    output_path="path/to/split_output.gpkg",
    split_polygons=True  # Enable pre-splitting for very large files
)

# Step 2: Aggregate results by original polygon IDs
final_results = aggregate_by_field(
    input_path="path/to/split_output.gpkg",
    output_path="path/to/final_output.gpkg",
    field_name="original_id"  # Field containing original polygon identifiers
)
```
### Calculate Vegetation Cover Ratios and Areas

```bash
from vCcCpy import get_VCratio, get_VCarea

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
## Test Scripts

The package includes comprehensive test scripts in the test directory:

- `test_01aa.py`: Chunkwise processing without pre-splitting using `get_VEGETATION()`
- `test_01a.py` + `test_01b.py`: Pre-splitting with `get_VEGETATION()` and feature aggregation with `aggregate_by_field()`
- `test_02.py`: Vegetation ratio and area computation with `get_VCratio()` and `get_VCarea()`

## Output

The package generates output files in various geospatial formats (GeoPackage, GeoJSON, Shapefile) containing:

- Original polygon geometries
- Calculated vegetation cover ratios (VCr)
- Calculated vegetation cover areas (VCa)
- Processing metadata and confidence metrics

## Memory Management

For optimal performance:

- Use default chunkwise processing for most cases
- Only enable pre-splitting for extremely large datasets (>1GB)
- Ensure adequate RAM availability for processing
- Monitor progress through the built-in progress bars

Happy coding! 🌿
