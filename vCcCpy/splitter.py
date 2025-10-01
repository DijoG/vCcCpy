import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import split
import math
from typing import List, Optional, Dict, Union
import logging

logger = logging.getLogger(__name__)

class AdaptivePolygonSplitter:
    """
    Adaptive polygon splitter for handling large polygons efficiently.
    """
    
    def __init__(self, n_areas: int = 8, min_area_ratio: float = 0.01):
        self.n_areas = n_areas
        self.min_area_ratio = min_area_ratio
    
    def split_polygon(self, polygon: Polygon, id_value: str, 
                     id_field: str = 'id') -> gpd.GeoDataFrame:
        """
        Split a single polygon into smaller sub-polygons.
        
        Parameters
        ----------
        polygon : shapely.Polygon
            Input polygon to split
        id_value : str
            Original polygon ID
        id_field : str
            ID field name
            
        Returns
        -------
        geopandas.GeoDataFrame
            Split polygons with attributes
        """
        if not isinstance(polygon, (Polygon, MultiPolygon)):
            raise ValueError("Input must be a Polygon or MultiPolygon")
        
        # For very small polygons, don't split
        area = polygon.area
        if area < 1000:  # 1000 m² threshold
            return gpd.GeoDataFrame(
                {id_field: [id_value], 'PolyArea': [area]},
                geometry=[polygon],
                crs=None
            )
        
        # Calculate optimal grid dimensions based on aspect ratio
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        aspect_ratio = width / height
        
        if aspect_ratio > 3:
            # Long and narrow
            grid_x = max(2, math.ceil(math.sqrt(self.n_areas * aspect_ratio)))
            grid_y = max(2, math.ceil(math.sqrt(self.n_areas / aspect_ratio)))
        elif aspect_ratio < 0.33:
            # Tall and narrow
            grid_x = max(2, math.ceil(math.sqrt(self.n_areas * aspect_ratio)))
            grid_y = max(2, math.ceil(math.sqrt(self.n_areas / aspect_ratio)))
        else:
            # Roughly square
            grid_x = max(2, math.ceil(math.sqrt(self.n_areas)))
            grid_y = max(2, math.ceil(math.sqrt(self.n_areas)))
        
        logger.debug(f"Splitting polygon with {grid_x}x{grid_y} grid")
        
        # Create grid and intersect with polygon
        grid_polygons = self._create_grid(bounds, grid_x, grid_y)
        split_polygons = self._intersect_with_polygon(grid_polygons, polygon)
        
        # Filter out tiny slivers
        area_threshold = max([p.area for p in split_polygons]) * self.min_area_ratio
        filtered_polygons = [p for p in split_polygons if p.area >= area_threshold]
        
        # Create result GeoDataFrame
        result_gdf = gpd.GeoDataFrame(
            {
                id_field: [id_value] * len(filtered_polygons),
                'PolyArea': [p.area for p in filtered_polygons]
            },
            geometry=filtered_polygons,
            crs=None
        )
        
        return result_gdf
    
    def _create_grid(self, bounds: tuple, grid_x: int, grid_y: int) -> List[Polygon]:
        """Create a grid of polygons covering the bounds."""
        minx, miny, maxx, maxy = bounds
        width = maxx - minx
        height = maxy - miny
        
        grid_polygons = []
        for i in range(grid_x):
            for j in range(grid_y):
                x1 = minx + (i * width / grid_x)
                x2 = minx + ((i + 1) * width / grid_x)
                y1 = miny + (j * height / grid_y)
                y2 = miny + ((j + 1) * height / grid_y)
                
                grid_polygons.append(box(x1, y1, x2, y2))
        
        return grid_polygons
    
    def _intersect_with_polygon(self, grid_polygons: List[Polygon], 
                               polygon: Polygon) -> List[Polygon]:
        """Intersect grid polygons with target polygon."""
        result_polygons = []
        
        for grid_poly in grid_polygons:
            try:
                intersection = grid_poly.intersection(polygon)
                if not intersection.is_empty:
                    if intersection.geom_type == 'Polygon':
                        result_polygons.append(intersection)
                    elif intersection.geom_type == 'MultiPolygon':
                        result_polygons.extend(list(intersection.geoms))
            except Exception as e:
                logger.warning(f"Intersection failed: {e}")
                continue
        
        return result_polygons

# Default splitting strategies for common categories
DEFAULT_SPLITTING_STRATEGIES = {
    "Wadis": {"threshold": 5000000, "n_areas": 40},      # Wadi features
    "Large_Parks": {"threshold": 200000, "n_areas": 20}, # Large parks
    "default": {"threshold": 2000000, "n_areas": 10}     # default
}

def split_large_polygons(gdf: gpd.GeoDataFrame, 
                        split_threshold: float = 2000000,
                        n_areas: int = 10,
                        id_field: str = 'id',
                        category_field: Optional[str] = None,
                        splitting_strategies: Optional[Dict[str, Dict]] = None) -> gpd.GeoDataFrame:
    """
    Split large polygons in a GeoDataFrame with category-based strategies.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons
    split_threshold : float
        Default area threshold for splitting (m²)
    n_areas : int
        Default target number of sub-polygons
    id_field : str
        ID field name
    category_field : str, optional
        Field name containing category information (e.g., 'MCAT')
    splitting_strategies : dict, optional
        Dictionary mapping categories to splitting strategies.
        Format: {'Category': {'threshold': value, 'n_areas': value}}
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with large polygons split according to strategies
    """
    # Calculate areas if not present
    if 'PolyArea' not in gdf.columns:
        gdf['PolyArea'] = gdf.geometry.area
    
    # Use default strategies if none provided
    if splitting_strategies is None:
        splitting_strategies = DEFAULT_SPLITTING_STRATEGIES
    
    # If no category field, use default strategy for all polygons
    if category_field is None or category_field not in gdf.columns:
        logger.info("No category field provided, using default splitting strategy for all polygons")
        return _split_with_single_strategy(gdf, split_threshold, n_areas, id_field)
    
    # Process by category
    logger.info(f"Using category-based splitting with field: {category_field}")
    
    # Get unique categories
    categories = gdf[category_field].unique()
    logger.info(f"Found categories: {list(categories)}")
    
    # Split results for each category
    split_results = []
    
    for category in categories:
        category_gdf = gdf[gdf[category_field] == category].copy()
        
        # Get strategy for this category
        strategy = splitting_strategies.get(category, splitting_strategies.get('default'))
        if strategy is None:
            strategy = {'threshold': split_threshold, 'n_areas': n_areas}
        
        cat_threshold = strategy.get('threshold', split_threshold)
        cat_n_areas = strategy.get('n_areas', n_areas)
        
        logger.info(f"Processing category '{category}': threshold={cat_threshold}, n_areas={cat_n_areas}")
        
        # Split polygons for this category
        category_split = _split_with_single_strategy(
            category_gdf, cat_threshold, cat_n_areas, id_field
        )
        
        split_results.append(category_split)
    
    # Combine all results
    if split_results:
        result_gdf = gpd.GeoDataFrame(
            pd.concat(split_results, ignore_index=True),
            crs=gdf.crs
        )
    else:
        result_gdf = gdf
    
    return result_gdf

def _split_with_single_strategy(gdf: gpd.GeoDataFrame,
                              split_threshold: float,
                              n_areas: int,
                              id_field: str) -> gpd.GeoDataFrame:
    """
    Split polygons using a single strategy (helper function).
    """
    # Separate large and small polygons
    large_polys = gdf[gdf['PolyArea'] > split_threshold].copy()
    small_polys = gdf[gdf['PolyArea'] <= split_threshold].copy()
    
    if len(large_polys) == 0:
        return gdf
    
    logger.info(f"Splitting {len(large_polys)} large polygons (threshold: {split_threshold}m²)")
    
    # Initialize splitter
    splitter = AdaptivePolygonSplitter(n_areas=n_areas)
    
    # Split large polygons
    split_results = []
    for idx, row in large_polys.iterrows():
        try:
            split_gdf = splitter.split_polygon(
                row.geometry, 
                row[id_field], 
                id_field
            )
            # Copy all other attributes to split polygons
            for col in row.index:
                if col not in [id_field, 'geometry', 'PolyArea'] and col in split_gdf.columns:
                    split_gdf[col] = row[col]
            split_results.append(split_gdf)
        except Exception as e:
            logger.error(f"Failed to split polygon {row[id_field]}: {e}")
            # Keep original polygon if splitting fails
            failed_gdf = gpd.GeoDataFrame(
                {id_field: [row[id_field]], 'PolyArea': [row['PolyArea']]},
                geometry=[row.geometry],
                crs=gdf.crs
            )
            # Copy other attributes
            for col in row.index:
                if col not in [id_field, 'geometry', 'PolyArea']:
                    failed_gdf[col] = row[col]
            split_results.append(failed_gdf)
    
    # Combine results with proper CRS handling
    if split_results:
        # Combine all split results without CRS first
        combined_split = pd.concat(split_results, ignore_index=True)
        
        # Create GeoDataFrame and set CRS explicitly
        large_split = gpd.GeoDataFrame(combined_split)
        large_split = large_split.set_crs(gdf.crs, allow_override=True)
        
        # Combine with small polygons
        if len(small_polys) > 0:
            combined_all = pd.concat([small_polys, large_split], ignore_index=True)
            result_gdf = gpd.GeoDataFrame(combined_all, crs=gdf.crs)
        else:
            result_gdf = large_split
    else:
        result_gdf = gdf
    
    return result_gdf

def create_splitting_strategy(categories: List[str], 
                            thresholds: List[float],
                            n_areas_list: List[int]) -> Dict[str, Dict]:
    """
    Create a splitting strategy dictionary from lists.
    
    Parameters
    ----------
    categories : list of str
        List of category names
    thresholds : list of float
        List of area thresholds for each category
    n_areas_list : list of int
        List of n_areas values for each category
        
    Returns
    -------
    dict
        Splitting strategy dictionary
    """
    if len(categories) != len(thresholds) or len(categories) != len(n_areas_list):
        raise ValueError("All input lists must have the same length")
    
    strategy = {}
    for cat, threshold, n_areas in zip(categories, thresholds, n_areas_list):
        strategy[cat] = {'threshold': threshold, 'n_areas': n_areas}
    
    return strategy

def analyze_polygon_sizes(gdf: gpd.GeoDataFrame, 
                         category_field: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze polygon sizes by category to help define splitting strategies.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons
    category_field : str, optional
        Field name containing category information
        
    Returns
    -------
    pandas.DataFrame
        Statistics about polygon sizes by category
    """
    # Calculate areas if not present
    if 'PolyArea' not in gdf.columns:
        gdf = gdf.copy()
        gdf['PolyArea'] = gdf.geometry.area
    
    if category_field and category_field in gdf.columns:
        # Group by category
        stats = gdf.groupby(category_field)['PolyArea'].agg([
            'count', 'min', 'max', 'mean', 'median', 'std'
        ]).round(2)
        
        # Add large polygon count (over 1km²)
        large_count = gdf[gdf['PolyArea'] > 1000000].groupby(category_field).size()
        stats['large_count'] = large_count
        stats['large_count'] = stats['large_count'].fillna(0).astype(int)
        
        # Add recommended threshold (75th percentile)
        threshold_rec = gdf.groupby(category_field)['PolyArea'].quantile(0.75).round(0)
        stats['recommended_threshold'] = threshold_rec
        
    else:
        # Overall statistics
        stats = pd.DataFrame({
            'count': [len(gdf)],
            'min': [gdf['PolyArea'].min()],
            'max': [gdf['PolyArea'].max()],
            'mean': [gdf['PolyArea'].mean()],
            'median': [gdf['PolyArea'].median()],
            'std': [gdf['PolyArea'].std()],
            'large_count': [len(gdf[gdf['PolyArea'] > 1000000])],
            'recommended_threshold': [gdf['PolyArea'].quantile(0.75)]
        }).round(2)
    
    return stats

# Export main functions
__all__ = [
    'AdaptivePolygonSplitter',
    'split_large_polygons',
    'create_splitting_strategy',
    'analyze_polygon_sizes',
    'DEFAULT_SPLITTING_STRATEGIES'
]