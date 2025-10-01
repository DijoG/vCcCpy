# Vegetation Canopy Cover Calculator in Python 3.13 (vCcCpy)

[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for calculating vegetation and/or canopy cover ratios from high resolution raster (10m or >10m) data for polygon features. Supports processing of large polygons through intelligent splitting and memory-efficient chunkwise processing.

## Features

- **Memory Efficient**: Handles large datasets through chunk processing and optimized memory usage
- **Intelligent Splitting**: Automatically splits large polygons to prevent memory issues
- **Multiple Outputs**: Calculate both vegetation cover ratios (VCr) and areas (VCa)
- **Flexible Input**: Supports various raster and vector formats
- **Progress Tracking**: Real-time progress monitoring with tqdm
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Installation

```bash
## Version 1:
pip install git+https://github.com/DijoG/vCcCpy.git

## Version 2:
# Clone the repository
git clone https://github.com/DijoG/vCcCpy.git
cd vCcCpy

# Install in development mode
pip install -e .
```

## Usage

Go to the 'test' directory and consider *test_01a.py*, *test_01b.py* for ***get_VEGETATION()*** and *test_02.py* for ***get_VCratio()*** and ***get_VCarea()***. 

## Important

- *****get_VEGETATION()*** was developed and tested using 0.3m resolution raster data.**
- *****get_VCratio()*** and ***get_VCarea()*** were developed and tested using 10m resolution raster data.**
- **Please NEVER change the code of the scripts present in this repository!**

***Have fun and happy coding!***
