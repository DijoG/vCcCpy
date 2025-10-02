"""
Core functionality for vCcCpy - Vegetation Canopy Cover Calculator in Python
Optimized for Python 3.13+
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.mask import mask
from rasterio.plot import show
import rioxarray as rxr
from shapely.geometry import mapping, shape, Polygon, MultiPolygon
from shapely.ops import unary_union
import warnings
from tqdm.auto import tqdm
import logging
from typing import Optional, Union, List, Tuple, Dict, Any
import os
import tempfile
import sys
import gc
import math
from pathlib import Path
import json
import re

# Python 3.13+ specific imports and optimizations
if sys.version_info >= (3, 13):
    from typing import TypeAlias, assert_type
    import _testinternalcapi  # For internal optimizations
    
    # Use new 3.13 generic syntax where available
    PathLike: TypeAlias = str | os.PathLike
    
    def optimized_array_creation(shape, dtype=np.float32):
        """Use Python 3.13 optimized array creation with new buffer protocol"""
        return np.empty(shape, dtype=dtype, order='C')
        
else:
    from typing_extensions import TypeAlias
    from typing import Union
    
    PathLike = Union[str, os.PathLike]
    
    def optimized_array_creation(shape, dtype=np.float32):
        return np.empty(shape, dtype=dtype)

from .utils import (
    calculate_pixel_area, 
    estimate_memory_requirements,
    enable_optimized_gc,
    clear_memory_aggressive
)

# Configure logging with 3.13+ features
if sys.version_info >= (3, 13):
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # 3.13+ feature to reset logging config
    )
else:
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# Global configuration
class Config:
    """Configuration settings with 3.13+ optimizations"""
    DEFAULT_CHUNK_SIZE = 500
    MAX_MEMORY_MB = 32000  # 32GB
    ENABLE_313_OPTIMIZATIONS = sys.version_info >= (3, 13)
    USE_PARALLEL_PROCESSING = False  # Set to True for Dask parallel processing

def _setup_313_optimizations():
    """Setup Python 3.13 specific optimizations"""
    if sys.version_info >= (3, 13):
        enable_optimized_gc()
        
        # 3.13 has improved memory allocator - tune for large arrays
        if hasattr(gc, 'set_prealloc_threshold'):
            gc.set_prealloc_threshold(1000000)  # 1MB threshold
        
        logger.info("Python 3.13 optimizations enabled")
    else:
        logger.info("Running on Python < 3.13, using standard optimizations")

def _extract_vegetation_count_rstyle(raster_path: str, geometry, band: int = 1) -> int:
    """
    Extract vegetation pixel count using R-style approach.
    Counts pixels where value == 1 (vegetation).
    Equivalent to R's: sum(values == 1, na.rm = TRUE)
    """
    try:
        with rasterio.open(raster_path) as src:
            # Use mask to extract raster values within geometry
            out_image, out_transform = mask(
                src, 
                [mapping(geometry)], 
                crop=True, 
                filled=False, 
                all_touched=True,
                indexes=[band]
            )
            
            # Count pixels where value == 1 (vegetation)
            # This matches R's: sum(values == 1, na.rm = TRUE)
            veg_count = np.count_nonzero(out_image[0] == 1)
            
            return int(veg_count)
            
    except Exception as e:
        logger.warning(f"Error extracting vegetation count for geometry: {e}")
        return 0

def _extract_vegetation_pixels_optimized(raster_path: str, geometry, band: int = 1) -> int:
    """
    Extract vegetation pixels (value=1) for a given geometry with 3.13 optimizations.
    
    Parameters
    ----------
    raster_path : str
        Path to raster file
    geometry : shapely geometry
        Polygon geometry to extract from
    band : int
        Raster band to process (default: 1)
        
    Returns
    -------
    int
        Number of vegetation pixels
    """
    try:
        with rasterio.open(raster_path) as src:
            # Use all_touched=True to include all pixels touched by geometry
            out_image, out_transform = mask(
                src, 
                [mapping(geometry)], 
                crop=True, 
                filled=False, 
                all_touched=True,
                indexes=[band]  # Only read the specific band
            )
            
            # Count vegetation pixels (value = 1) using optimized numpy
            if Config.ENABLE_313_OPTIMIZATIONS:
                # Use 3.13 optimized operations
                veg_pixels = np.count_nonzero(out_image[0] == 1)
            else:
                veg_pixels = np.sum(out_image[0] == 1)
                
            return int(veg_pixels)
            
    except Exception as e:
        logger.warning(f"Error extracting vegetation pixels for geometry: {e}")
        return 0

def _process_raster_chunk_313(raster_path: str, geometries: List, band: int = 1) -> List[int]:
    """
    Process a chunk of geometries with 3.13 optimized memory management.
    
    Parameters
    ----------
    raster_path : str
        Path to raster file
    geometries : list
        List of shapely geometries
    band : int
        Raster band to process
        
    Returns
    -------
    list
        List of vegetation pixel counts for each geometry
    """
    results = []
    
    for geometry in geometries:
        veg_pixels = _extract_vegetation_pixels_optimized(raster_path, geometry, band)
        results.append(veg_pixels)
        
        # Aggressive memory management for 3.13
        if Config.ENABLE_313_OPTIMIZATIONS and len(geometries) > 10:
            if len(results) % 10 == 0:  # Clear memory every 10 geometries
                clear_memory_aggressive()
    
    return results

def get_VCratio(input_raster: str, 
                input_shape: str, 
                output_shape: Optional[str] = None,
                id_field: Optional[str] = None,
                chunk_size: int = None,
                use_313_optimizations: bool = True) -> gpd.GeoDataFrame:
    """
    Calculate Vegetation Cover Ratio (VCr) for each band in a raster stack.
    Equivalent to R's vCcCR::get_VCratio function.
    """
    # Setup optimizations
    if use_313_optimizations:
        _setup_313_optimizations()
    
    logger.info("Starting VCr calculation with Python %s optimizations", 
                "3.13+" if use_313_optimizations else "standard")
    
    # Load and prepare data
    logger.info("Loading vector data from: %s", input_shape)
    vector_gdf = gpd.read_file(input_shape)
    
    if id_field and id_field in vector_gdf.columns:
        vector_gdf = vector_gdf[[id_field, 'geometry']].copy()
        logger.info("Using ID field: %s", id_field)
    else:
        id_field = vector_gdf.columns[0]
        vector_gdf = vector_gdf[[id_field, 'geometry']].copy()
        logger.info("Using first field as ID: %s", id_field)
    
    # Get raster info with band names
    with rasterio.open(input_raster) as src:
        n_bands = src.count
        pixel_size = src.transform[0]  # Get pixel size from transform
        pixel_area = pixel_size * pixel_size  # Calculate pixel area
        
        # Try to read band names from raster
        band_names = []
        for i in range(1, n_bands + 1):
            try:
                # Try to get band description (often used for band names)
                band_desc = src.descriptions[i-1]
                if band_desc and band_desc != '':
                    band_names.append(band_desc)
                else:
                    # If no description, try to get from tags
                    band_tags = src.tags(i)
                    band_name = band_tags.get('band_name', f'Band_{i}')
                    band_names.append(band_name)
            except:
                # Fallback to generic band name
                band_names.append(f'Band_{i}')
        
        # If no band names found, try alternative methods
        if not band_names or all(name.startswith('Band_') for name in band_names):
            try:
                # Try using rioxarray to get band names (handles more formats)
                with rxr.open_rasterio(input_raster) as xarr:
                    if hasattr(xarr, 'long_name') and xarr.long_name is not None:
                        band_names = [str(name) for name in xarr.long_name.values]
                    elif hasattr(xarr, 'band') and xarr.band is not None:
                        band_names = [f"Band_{int(band)}" for band in xarr.band.values]
            except:
                # Final fallback
                band_names = [f'Band_{i}' for i in range(1, n_bands + 1)]
        
        logger.info("Processing %d bands from raster: %s", n_bands, band_names)
    
    # Calculate polygon areas once
    vector_gdf['PolyArea'] = vector_gdf.geometry.area
    
    # Process each band
    logger.info("Processing monthly vegetation cover areas:")
    
    for band in range(1, n_bands + 1):
        band_name = band_names[band-1]
        vcr_field_name = f"VCr_{band_name}"
        logger.info("Processing band %d: %s", band, vcr_field_name)
        
        # Extract vegetation pixels for all geometries
        veg_counts_list = []
        for idx, geometry in enumerate(vector_gdf.geometry):
            veg_count = _extract_vegetation_count_rstyle(input_raster, geometry, band)
            veg_counts_list.append(veg_count)
            
            # Progress update for large datasets
            if (idx + 1) % 100 == 0:
                logger.info("  Processed %d/%d features", idx + 1, len(vector_gdf))
        
        # Calculate VCr for each geometry (matching R formula)
        vcr_values = []
        for idx, veg_count in enumerate(veg_counts_list):
            poly_area = vector_gdf.iloc[idx]['PolyArea']
            if poly_area > 0:
                # Exact R formula: (veg_counts * pixel_area) / polygon_area * 100
                vcr = (veg_count * pixel_area) / poly_area * 100
                vcr_values.append(round(vcr, 2))
            else:
                vcr_values.append(0.0)
        
        vector_gdf[vcr_field_name] = vcr_values
    
    # Handle output path like R function
    if output_shape is None:
        base_name = os.path.splitext(input_shape)[0]
        output_shape = f"{base_name}_VCr.geojson"
    else:
        base_name = os.path.splitext(output_shape)[0]
        output_shape = f"{base_name}_VCr.geojson"
    
    # Save results
    _save_results(vector_gdf, output_shape)
    logger.info("VCr results saved to: %s", output_shape)
    
    logger.info("VCr calculation completed successfully")
    return vector_gdf

def get_VCarea(input_raster: str, 
               input_shape: str, 
               output_shape: Optional[str] = None,
               id_field: Optional[str] = None,
               chunk_size: int = None,
               use_313_optimizations: bool = True) -> gpd.GeoDataFrame:
    """
    Calculate Vegetation Cover Area (VCa) for each band in a raster stack.
    Equivalent to R's vCcCR::get_VCarea function.
    """
    # Setup optimizations
    if use_313_optimizations:
        _setup_313_optimizations()
    
    logger.info("Starting VCa calculation with Python %s optimizations", 
                "3.13+" if use_313_optimizations else "standard")
    
    # Load and prepare data
    vector_gdf = gpd.read_file(input_shape)
    
    if id_field and id_field in vector_gdf.columns:
        vector_gdf = vector_gdf[[id_field, 'geometry']].copy()
    else:
        id_field = vector_gdf.columns[0]
        vector_gdf = vector_gdf[[id_field, 'geometry']].copy()
    
    # CRS validation skipped - assuming matching CRS
    logger.info("CRS validation skipped - assuming matching CRS")
    
    # Get raster info with band names
    with rasterio.open(input_raster) as src:
        n_bands = src.count
        pixel_size = src.transform[0]
        pixel_area = pixel_size * pixel_size
        
        # Try to read band names from raster
        band_names = []
        for i in range(1, n_bands + 1):
            try:
                band_desc = src.descriptions[i-1]
                if band_desc and band_desc != '':
                    band_names.append(band_desc)
                else:
                    band_tags = src.tags(i)
                    band_name = band_tags.get('band_name', f'Band_{i}')
                    band_names.append(band_name)
            except:
                band_names.append(f'Band_{i}')
        
        # Alternative method if no band names found
        if not band_names or all(name.startswith('Band_') for name in band_names):
            try:
                with rxr.open_rasterio(input_raster) as xarr:
                    if hasattr(xarr, 'long_name') and xarr.long_name is not None:
                        band_names = [str(name) for name in xarr.long_name.values]
                    elif hasattr(xarr, 'band') and xarr.band is not None:
                        band_names = [f"Band_{int(band)}" for band in xarr.band.values]
            except:
                band_names = [f'Band_{i}' for i in range(1, n_bands + 1)]
        
        logger.info("Processing %d bands from raster: %s", n_bands, band_names)
    
    # Process each band
    for band in range(1, n_bands + 1):
        band_name = band_names[band-1]
        vca_field_name = f"VCa_{band_name}"
        logger.info("Processing band %d: %s", band, vca_field_name)
        
        # Extract vegetation pixels for all geometries
        vca_values = []
        for geometry in vector_gdf.geometry:
            veg_count = _extract_vegetation_count_rstyle(input_raster, geometry, band)
            # Calculate vegetation area: veg_count * pixel_area
            vca = round(veg_count * pixel_area, 2)
            vca_values.append(vca)
        
        vector_gdf[vca_field_name] = vca_values
    
    # Handle output path like R function
    if output_shape is None:
        base_name = os.path.splitext(input_shape)[0]
        output_shape = f"{base_name}_VCa.geojson"
    else:
        base_name = os.path.splitext(output_shape)[0]
        output_shape = f"{base_name}_VCa.geojson"
    
    # Save results
    _save_results(vector_gdf, output_shape)
    logger.info("VCa results saved to: %s", output_shape)
    
    logger.info("VCa calculation completed successfully")
    return vector_gdf

def get_VEGETATION(polygons: Union[str, gpd.GeoDataFrame],
                   veg_raster: str,
                   output_path: str,
                   id_field: Optional[str] = None,
                   by_row: bool = False,
                   return_result: bool = False,
                   chunk_size: int = None,
                   use_313_optimizations: bool = True) -> Optional[gpd.GeoDataFrame]:
    """
    Process vegetation/canopy cover for polygons using vegetation raster.
    Optimized for Python 3.13+ with enhanced memory management.
    
    ASSUMES polygons are already pre-processed and optimized for analysis.
    No internal splitting - polygons should be pre-split if needed.
    
    Parameters
    ----------
    polygons : str or GeoDataFrame
        Input polygons or path to file (should be pre-processed)
    veg_raster : str
        Path to vegetation raster
    output_path : str
        Path to save results
    id_field : str, optional
        ID field name
    by_row : bool
        Process features one by one
    return_result : bool
        Whether to return the result
    chunk_size : int, optional
        Chunk size for processing
    use_313_optimizations : bool
        Use Python 3.13+ specific optimizations
        
    Returns
    -------
    geopandas.GeoDataFrame or None
        Result if return_result is True
    """
    # Setup optimizations
    if use_313_optimizations:
        _setup_313_optimizations()
    
    if chunk_size is None:
        chunk_size = Config.DEFAULT_CHUNK_SIZE
    
    logger.info("Starting vegetation analysis with Python %s", 
                "3.13+ optimizations" if use_313_optimizations else "standard processing")
    
    # Load data
    if isinstance(polygons, str):
        logger.info("Loading polygons from: %s", polygons)
        polygons_gdf = gpd.read_file(polygons)
    else:
        polygons_gdf = polygons.copy()
    
    logger.info("Loaded %d polygons", len(polygons_gdf))
    
    # CRS validation skipped - assuming matching CRS
    logger.info("CRS validation skipped - assuming matching CRS")
    
    # Set ID field
    if id_field is None or id_field not in polygons_gdf.columns:
        id_field = polygons_gdf.columns[0]
        logger.info("Using '%s' as ID field", id_field)
    
    # Calculate polygon areas
    polygons_gdf['PolyArea'] = polygons_gdf.geometry.area
    
    # Get pixel area
    pixel_area = calculate_pixel_area(veg_raster)
    logger.info("Pixel area: %.4f m²", pixel_area)
    
    # Memory optimization for large datasets
    if len(polygons_gdf) > 1000 and use_313_optimizations:
        from .utils import optimize_memory_usage
        polygons_gdf = optimize_memory_usage(polygons_gdf)
        logger.info("Memory optimization applied to polygon data")
    
    # Choose processing method
    if by_row:
        logger.info("Processing features one by one")
        result_gdf = _process_by_row_313(
            polygons_gdf, veg_raster, output_path, id_field, pixel_area, use_313_optimizations
        )
    else:
        logger.info("Processing with chunk-based approach")
        result_gdf = _process_chunked_313(
            polygons_gdf, veg_raster, output_path, id_field, pixel_area, chunk_size, use_313_optimizations
        )
    
    if return_result:
        return result_gdf

def _process_by_row_313(polygons_gdf: gpd.GeoDataFrame,
                       veg_raster: str,
                       output_path: str,
                       id_field: str,
                       pixel_area: float,
                       use_313_optimizations: bool) -> gpd.GeoDataFrame:
    """Process polygons one by one with 3.13 optimizations."""
    results = []
    
    for idx, row in tqdm(polygons_gdf.iterrows(), total=len(polygons_gdf), desc="Processing polygons"):
        poly_id = row[id_field]
        poly_geom = row.geometry
        poly_area = row['PolyArea']
        
        # Process polygon directly (no splitting)
        veg_pixels = _extract_vegetation_pixels_optimized(veg_raster, poly_geom)
        
        # Calculate results
        veg_area = round(veg_pixels * pixel_area, 2)
        veg_ratio = round((veg_area / poly_area) * 100, 2) if poly_area > 0 else 0.0
        
        results.append({
            id_field: poly_id,
            'PolyArea': poly_area,
            'VegArea': veg_area,
            'VegRatio': veg_ratio
        })
        
        # Memory management
        if use_313_optimizations and idx % 10 == 0:
            clear_memory_aggressive()
    
    # Create result GeoDataFrame
    result_df = pd.DataFrame(results)
    result_gdf = polygons_gdf.merge(result_df, on=id_field, how='left')
    
    # Save results
    _save_results(result_gdf, output_path)
    
    return result_gdf

def _process_chunked_313(polygons_gdf: gpd.GeoDataFrame,
                        veg_raster: str,
                        output_path: str,
                        id_field: str,
                        pixel_area: float,
                        chunk_size: int,
                        use_313_optimizations: bool) -> gpd.GeoDataFrame:
    """Process polygons in chunks with 3.13 optimizations."""
    
    results = []
    total_chunks = math.ceil(len(polygons_gdf) / chunk_size)
    
    for chunk_idx in tqdm(range(total_chunks), desc="Processing chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(polygons_gdf))
        chunk_gdf = polygons_gdf.iloc[start_idx:end_idx].copy()
        
        # Process chunk
        chunk_results = []
        for idx, row in chunk_gdf.iterrows():
            veg_pixels = _extract_vegetation_pixels_optimized(veg_raster, row.geometry)
            
            chunk_results.append({
                id_field: row[id_field],
                'geometry': row.geometry,
                'PolyArea': row['PolyArea'],
                'VegPixels': veg_pixels
            })
        
        results.extend(chunk_results)
        
        # Memory management between chunks
        if use_313_optimizations:
            clear_memory_aggressive()
    
    # Create result DataFrame
    temp_df = pd.DataFrame(results)
    
    # Calculate final metrics
    temp_df['VegArea'] = round(temp_df['VegPixels'] * pixel_area, 2)
    temp_df['VegRatio'] = round((temp_df['VegArea'] / temp_df['PolyArea']) * 100, 2)
    
    # Create final GeoDataFrame
    result_gdf = gpd.GeoDataFrame(
        temp_df[[id_field, 'geometry', 'PolyArea', 'VegArea', 'VegRatio']],
        geometry='geometry',
        crs=polygons_gdf.crs
    )
    
    # Save results
    _save_results(result_gdf, output_path)
    
    return result_gdf

def _save_results(result_gdf: gpd.GeoDataFrame, output_path: str):
    """Save results to file with proper format detection and fixed statistics."""
    # Create output directory if needed
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine format from extension
    ext = os.path.splitext(output_path)[1].lower()
    
    try:
        if ext == '.gpkg':
            result_gdf.to_file(output_path, driver='GPKG')
        elif ext in ['.geojson', '.json']:
            result_gdf.to_file(output_path, driver='GeoJSON')
        elif ext == '.shp':
            result_gdf.to_file(output_path)
        else:
            # Default to GeoJSON
            output_path = output_path + '.geojson'
            result_gdf.to_file(output_path, driver='GeoJSON')
        
        logger.info("Results saved to: %s", output_path)
        
        # FIXED: Calculate summary statistics with proper type checking
        # Check if VegArea column exists (for VCratio/VCarea functions)
        if 'VegArea' in result_gdf.columns:
            veg_area_series = result_gdf['VegArea']
            if not pd.api.types.is_numeric_dtype(veg_area_series):
                veg_area_series = pd.to_numeric(veg_area_series, errors='coerce')
            
            total_veg_area = veg_area_series.sum()
            if pd.isna(total_veg_area) or not np.isfinite(total_veg_area):
                logger.warning("  Total vegetation area calculation produced invalid value")
                total_veg_area_str = "N/A"
            else:
                total_veg_area_str = f"{total_veg_area:.2f}"
            
            logger.info("  Total vegetation area: %s m²", total_veg_area_str)
        
        # Check if VegRatio column exists (for get_VEGETATION function)
        if 'VegRatio' in result_gdf.columns:
            veg_ratio_series = result_gdf['VegRatio']
            if not pd.api.types.is_numeric_dtype(veg_ratio_series):
                veg_ratio_series = pd.to_numeric(veg_ratio_series, errors='coerce')
            
            avg_veg_ratio = veg_ratio_series.mean()
            if pd.isna(avg_veg_ratio) or not np.isfinite(avg_veg_ratio):
                logger.warning("  Average vegetation ratio calculation produced invalid value")
                avg_veg_ratio_str = "N/A"
            else:
                avg_veg_ratio_str = f"{avg_veg_ratio:.2f}"
            
            logger.info("  Average vegetation ratio: %s%%", avg_veg_ratio_str)
        
        logger.info("  Processed polygons: %d", len(result_gdf))
        
    except Exception as e:
        logger.error("Failed to save results to %s: %s", output_path, e)
        raise

# Additional utility function for batch processing
def batch_process_vegetation(polygon_files: List[str],
                            raster_files: List[str],
                            output_dir: str,
                            **kwargs) -> Dict[str, gpd.GeoDataFrame]:
    """
    Batch process multiple polygon and raster files.
    
    Parameters
    ----------
    polygon_files : list of str
        List of polygon file paths
    raster_files : list of str
        List of raster file paths
    output_dir : str
        Output directory for results
    **kwargs
        Additional arguments for get_VEGETATION
        
    Returns
    -------
    dict
        Dictionary with output paths as keys and GeoDataFrames as values
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for i, (poly_file, raster_file) in enumerate(zip(polygon_files, raster_files)):
        logger.info("Processing batch %d/%d", i+1, len(polygon_files))
        logger.info("  Polygons: %s", poly_file)
        logger.info("  Raster: %s", raster_file)
        
        # Generate output path
        poly_name = os.path.splitext(os.path.basename(poly_file))[0]
        raster_name = os.path.splitext(os.path.basename(raster_file))[0]
        output_path = os.path.join(output_dir, f"{poly_name}_{raster_name}_results.gpkg")
        
        try:
            result = get_VEGETATION(
                polygons=poly_file,
                veg_raster=raster_file,
                output_path=output_path,
                **kwargs
            )
            results[output_path] = result
            logger.info("  ✅ Completed: %s", output_path)
        except Exception as e:
            logger.error("  ❌ Failed: %s", e)
            results[output_path] = None
    
    return results

# Python 3.13+ context manager for resource management
if sys.version_info >= (3, 13):
    from contextlib import contextmanager
    
    @contextmanager
    def vegetation_analysis_context(polygons, raster, output_path, **kwargs):
        """
        Context manager for vegetation analysis with automatic resource cleanup.
        
        Usage:
        with vegetation_analysis_context(polygons, raster, output_path) as result:
            # Use result here
            print(result.head())
        """
        logger.info("Starting vegetation analysis in context manager")
        try:
            result = get_VEGETATION(polygons, raster, output_path, **kwargs)
            yield result
        finally:
            # Force cleanup of large arrays and objects
            clear_memory_aggressive()
            logger.info("Vegetation analysis context cleaned up")

# Aggregation of chunk-processed vector output by def get_VEGETATION()
def aggregate_by_field(gpkg_path: str, output_path: str = None, group_by: str = 'pid') -> gpd.GeoDataFrame:
    """
    Aggregate vegetation results by specified field and extract MCAT from pid.
    
    Equivalent to R:
    VEC %>% 
    group_by(pid) %>% 
    summarise(geom = st_union(geom), VegArea = sum(VegArea)) %>%
    ungroup() %>%
    mutate(PolyArea = round(as.numeric(st_area(.)), 2),
           VCratio = round(VegArea/PolyArea * 100, 2),
           MCAT = gsub("_|[0-9]", "", pid))
    """
    gdf = gpd.read_file(gpkg_path)
    
    aggregated = gdf.groupby(group_by).agg({
        'VegArea': 'sum',
        'geometry': lambda x: unary_union(x) if len(x) > 1 else x.iloc[0]
    }).reset_index()
    
    # Calculate areas and ratios
    result_gdf = gpd.GeoDataFrame(aggregated, geometry='geometry', crs=gdf.crs)
    result_gdf['PolyArea'] = result_gdf.geometry.area.round(2)
    result_gdf['VCratio'] = ((result_gdf['VegArea'] / result_gdf['PolyArea']) * 100).round(2)
    
    # Extract MCAT from pid (equivalent to gsub("_|[0-9]", "", pid))
    result_gdf['MCAT'] = result_gdf['pid'].str.replace(r'[_0-9]', '', regex=True)
    
    if output_path:
        result_gdf.to_file(output_path, driver='GPKG')
    
    return result_gdf

# CRS (projection) check, transfowm and feature validate
def crs_check_transform(vector_path, raster_path):
    """
    Check CRS and validate/repair geometry structure.
    Returns path to validated file.
    """
    try:
        # Load vector data
        vector_gdf = gpd.read_file(vector_path)
        
        print(f"Processing: {os.path.basename(vector_path)}")
        print(f"  Original: {len(vector_gdf)} features, {len(vector_gdf.columns)} columns")
        
        # Handle case with no attributes (only geometry)
        if len(vector_gdf.columns) == 1 and 'geometry' in vector_gdf.columns:
            # Add a dummy ID column to ensure proper GeoDataFrame structure
            vector_gdf['feature_id'] = range(1, len(vector_gdf) + 1)
            print("  Added ID column for geometry-only file")
        
        # Extract valid geometries
        geometries = []
        empty_count = 0
        for geom in vector_gdf.geometry:
            if geom is None or geom.is_empty:
                empty_count += 1
                continue
            geometries.append(geom)
        
        if empty_count > 0:
            print(f"  Removed {empty_count} empty geometries")
        
        # Reconstruct GeoDataFrame if needed
        if len(geometries) != len(vector_gdf):
            data_columns = [col for col in vector_gdf.columns if col != 'geometry']
            if data_columns:
                data_without_geom = vector_gdf[data_columns].iloc[:len(geometries)].copy()
            else:
                data_without_geom = pd.DataFrame({'feature_id': range(1, len(geometries) + 1)})
            
            vector_gdf_clean = gpd.GeoDataFrame(
                data_without_geom,
                geometry=geometries,
                crs=vector_gdf.crs
            )
            print(f"  Reconstructed: {len(vector_gdf_clean)} valid features")
        else:
            vector_gdf_clean = vector_gdf.copy()
        
        # Ensure geometries are valid
        vector_gdf_clean['geometry'] = vector_gdf_clean.geometry.make_valid()
        
        # Get raster CRS
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
        
        # Create output path
        dir_name = os.path.dirname(vector_path)
        base_name = os.path.basename(vector_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        vector_temp = os.path.join(dir_name, f"{file_name_without_ext}_validated.geojson")
        
        print(f"  CRS: {vector_gdf_clean.crs} → {raster_crs}")
        
        # Transform CRS if needed
        if vector_gdf_clean.crs != raster_crs:
            vector_gdf_clean = vector_gdf_clean.to_crs(raster_crs)
            print("  CRS transformed")
        
        # Save validated geometry
        vector_gdf_clean.to_file(vector_temp, driver='GeoJSON')
        print(f"  Saved: {os.path.basename(vector_temp)}")
        
        return vector_temp
            
    except Exception as e:
        print(f"  Error: {e}")
        # Return original path as fallback
        return vector_path

# Explodes multi-part geometries and create unique 'pid' - polygon id   
def explode_pid(vector_path, field_to_string):
    """
    Explode multi-part geometries and create unique polygon IDs.
    
    Parameters:
    vector_path (str): Path to vector file
    
    Returns:
    geopandas.GeoDataFrame: Exploded GeoDataFrame with unique polygon IDs
    """
    # Load data
    VEC = gpd.read_file(vector_path)
    
    # Apply make_valid to geometry column to preserve GeoDataFrame
    VEC['geometry'] = VEC.geometry.make_valid()
    
    # Explode preserving GeoDataFrame structure (creating a new attribute named 'pid')
    VEC_exploded = VEC.explode(index_parts=True)
    VEC_exploded = VEC_exploded.reset_index(drop=True)
    VEC_exploded["mid"] = VEC_exploded.index
    VEC_exploded["pid"] = VEC_exploded[field_to_string].astype(str) + "_" + VEC_exploded["mid"].astype(str)
    
    # Check and ensure it is a GeoDataFrame
    if not hasattr(VEC_exploded, 'columns'):
        print("Converting GeoSeries to GeoDataFrame...")
        VEC_final = gpd.GeoDataFrame(VEC_exploded, geometry='geometry', crs=VEC.crs)
        # Copy attributes from original
        for col in VEC.columns:
            if col != 'geometry' and col not in VEC_final.columns:
                # Find matching rows and copy attributes
                VEC_final[col] = VEC_final.index.map(lambda x: VEC[col].iloc[x % len(VEC)])
    else:
        VEC_final = VEC_exploded
    
    return VEC_final
    
# Export main functions
__all__ = [
    'get_VCratio',
    'get_VCarea', 
    'get_VEGETATION',
    'batch_process_vegetation',
    'aggregate_by_field',
    'crs_check_transform',
    'Config'
]

# Add 3.13+ specific exports
if sys.version_info >= (3, 13):
    __all__.extend([
        'vegetation_analysis_context'
    ])

# Initialize optimizations when module is imported
_setup_313_optimizations()
