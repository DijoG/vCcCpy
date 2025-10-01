# Vegetation Canopy Cover Calculator in Python 3.13 (vCcCpy)

[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for calculating vegetation and/or canopy cover ratios from high resolution raster data for polygon features. Supports processing of large polygons through intelligent splitting and memory-efficient chunkwise processing.

## Features

- **Memory Efficient**: Handles large datasets through chunk processing and optimized memory usage
- **Intelligent Splitting**: Automatically splits large polygons to prevent memory issues
- **Multiple Outputs**: Calculate both vegetation cover ratios (VCr) and areas (VCa)
- **Flexible Input**: Supports various raster and vector formats
- **Progress Tracking**: Real-time progress monitoring with tqdm
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Installation

```bash
pip install vCcCpy
```
## Usage

Go to the 'test' directory and consider 'test_01a.py', 'test_01b.py' and 'test_02.py' files. Have fun and happy coding!
