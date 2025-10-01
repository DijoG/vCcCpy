"""
Vegetation Canopy Cover Calculator in Python (vCcCpy)

A Python package for calculating vegetation and/or canopy cover ratios 
from raster data for polygon features. Supports processing of large 
polygons through intelligent splitting and memory-efficient processing.
"""

__version__ = "0.1.0"
__author__ = "Gergo Dioszegi"

from .core import get_VCratio, get_VCarea, get_VEGETATION, aggregate_by_field, crs_check_transform
from .utils import calculate_pixel_area, optimize_memory_usage  # Removed validate_crs
from .splitter import AdaptivePolygonSplitter, split_large_polygons, analyze_polygon_sizes

__all__ = [
    'get_VCratio',
    'get_VCarea', 
    'get_VEGETATION',
    'aggregate_by_field',
    'crs_check_transform',
    'calculate_pixel_area',
    'optimize_memory_usage',
    'AdaptivePolygonSplitter',
    'split_large_polygons', 
    'analyze_polygon_sizes'
]