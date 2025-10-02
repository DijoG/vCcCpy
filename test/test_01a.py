import geopandas as gpd
import sys
import os
from vCcCpy.core import get_VEGETATION, explode_pid
from vCcCpy.splitter import split_large_polygons, analyze_polygon_sizes
import time

# Load data, explode preserving GeoDataFrame structure (creating a new attribute named 'pid' concatenating 'field_to_string')
GRP = explode_pid("D:/KPI/vector/GRPtypo.geojson", field_to_string='MCAT')

# Filter "Large_Parks" and "Wadis" in the 'MCAT' attribute for testing
GRP_test = GRP[GRP['MCAT'].isin(["Large_Parks", "Wadis"])].copy()   
print(f"Original filtered polygons: {len(GRP_test)}")
print(f"Categories: {GRP_test['MCAT'].unique().tolist()}")
print(f"Total area: {GRP_test.geometry.area.sum():.0f} m²")

# 1) Pre-splitting process -----------------------------------------------------------------
GRP_test['PolyArea'] = GRP_test.geometry.area

# Split the polygons
stats = analyze_polygon_sizes(GRP_test, category_field="MCAT")  # Use "MCAT" not "pid"
print("Polygon size statistics:")
print(stats)

# Then create strategies based on the analysis
strategies = {
    "Wadis": {"threshold": 5000000, "n_areas": 30},
    "Large_Parks": {"threshold": 200000, "n_areas": 20}  
    # "default": {"threshold": 200000, "n_areas": 10}          # for anything else not defined
}

# If you want to use stats:
# strategies = {
#     "Wadis": {"threshold": 2000000, "n_areas": 40},
#     "Large_Parks": {"threshold": stats.loc["Large_Parks", "recommended_threshold"], "n_areas": 20}
# }

GRP_pre_split = split_large_polygons(GRP_test, 
                                     category_field="MCAT", 
                                     splitting_strategies=strategies,
                                     id_field="pid")

print(f"After pre-splitting: {len(GRP_pre_split)} polygons")

# 2) Vegetation analysis -------------------------------------------------------------------
result = get_VEGETATION(
    polygons=GRP_pre_split,
    veg_raster=r"C:/.../1232-T2-TM2_1-GIS-Remote-Sensing/06_GIS-Data/09_LCC/LCC_2022_VC_CC/LCC_2022_4_1to1_CC_EPSG32638.tif",
    output_path=r"D:/KPI/vector/CC22test_LPWadioptimum.gpkg",
    id_field="pid",
    by_row=True,           
    return_result=True
)

print("Success!")
print(f"Output features: {len(result)}")
print(result[['pid', 'VegArea', 'VegRatio']].head())
# FINITO -----------------------------------------------------------------------------------