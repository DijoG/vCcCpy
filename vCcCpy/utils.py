"""
Utility functions for vCcCpy
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import logging
import gc
import os

logger = logging.getLogger(__name__)

def calculate_pixel_area(raster_path: str) -> float:
    """Calculate pixel area from raster."""
    with rasterio.open(raster_path) as src:
        transform = src.transform
        pixel_width = transform[0]
        pixel_height = -transform[4]  # Negative because transform[4] is usually negative
        return pixel_width * pixel_height

def optimize_memory_usage(gdf):
    """Optimize memory usage by downcasting numeric columns."""
    optimized_gdf = gdf.copy()
    
    for col in optimized_gdf.columns:
        if col == 'geometry':
            continue
            
        try:
            if optimized_gdf[col].dtype in ['float64', 'float32']:
                if optimized_gdf[col].isna().any() or np.isinf(optimized_gdf[col]).any():
                    optimized_gdf[col] = optimized_gdf[col].astype(np.float32)
                else:
                    optimized_gdf[col] = optimized_gdf[col].astype(np.uint16)
            elif optimized_gdf[col].dtype in ['int64', 'int32']:
                if optimized_gdf[col].isna().any():
                    optimized_gdf[col] = optimized_gdf[col].fillna(0).astype(np.uint16)
                else:
                    optimized_gdf[col] = optimized_gdf[col].astype(np.uint16)
        except Exception as e:
            continue
    
    return optimized_gdf

def estimate_memory_requirements(n_polygons, n_bands=1):
    """Estimate memory requirements for processing."""
    base_memory = n_polygons * n_bands * 100  # bytes per polygon per band
    return base_memory / (1024 * 1024)  # Convert to MB

def enable_optimized_gc():
    """Enable optimized garbage collection for Python 3.13+."""
    gc.enable()
    gc.set_threshold(700, 10, 10)

def clear_memory_aggressive():
    """Aggressive memory clearing."""
    if gc.isenabled():
        gc.collect()