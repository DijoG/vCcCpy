import geopandas as gpd
import sys
import os
from vCcCpy.core import get_VEGETATION, explode_pid
import time
import pandas as pd  

# 1) Data preparation ----------------------------------------------------------------------
# Load data, explode preserving GeoDataFrame structure (creating a new attribute named 'pid' concatenating the attribute defined in field_to_string)
GRP = explode_pid("D:/KPI/vector/GRPfullNAME.geojson", field_to_string="NAME_ENGLI")  # "D:/KPI/vector/GRPtypo.geojson" or "D:/KPI/vector/GRPfullNAME.geojson"; 'MCAT' or 'NAME_ENGLI'

# ADD DIAGNOSTICS: Check original data
print(f"Original data loaded: {len(GRP)} polygons")
print(f"Original categories: {GRP['CATEGORY'].unique()}")
original_categories = set(GRP['CATEGORY'].unique())
original_count = len(GRP)

# Add 'PolyArea' column 
GRP['PolyArea'] = GRP.geometry.area

# Rename data (no splitting -> preserving all features):
GRP_no_split = GRP

print("\n=== INTERMEDIATE DIAGNOSTICS ===")
print(f"Original count: {original_count}")
print(f"After pre-splitting: {len(GRP_no_split)}")
print(f"Net change: {len(GRP_no_split) - original_count} polygons")

# Check category preservation
if 'CATEGORY' in GRP_no_split.columns:
    split_categories = set(GRP_no_split['CATEGORY'].unique())
    print(f"Original categories: {original_categories}")
    print(f"After splitting categories: {split_categories}")
    
    missing_categories = original_categories - split_categories
    if missing_categories:
        print(f"⚠️  WARNING: Missing categories: {missing_categories}")
        
        # Debug missing categories
        for missing_cat in missing_categories:
            original_cat_count = len(GRP[GRP['CATEGORY'] == missing_cat])
            print(f"  {missing_cat}: had {original_cat_count} polygons originally")
            
            # Check if they were below splitting threshold
            cat_polygons = GRP[GRP['CATEGORY'] == missing_cat]
            max_area = cat_polygons['PolyArea'].max() if 'PolyArea' in cat_polygons.columns else cat_polygons.geometry.area.max()
            print(f"    Max area in category: {max_area:,.0f} m²")
    else:
        print("✅ All categories preserved")

# Check area preservation
original_total_area = GRP['PolyArea'].sum()
split_total_area = GRP_no_split['PolyArea'].sum()

# A nuanced area change reporting
area_diff_pct = ((split_total_area - original_total_area) / original_total_area) * 100
print(f"Area preservation: {area_diff_pct:.4f}% difference")

if abs(area_diff_pct) > 5.0:
    print("⚠️  WARNING: Significant area change detected!")
elif abs(area_diff_pct) > 1.0:
    print("ℹ️  Notice: Minor area change detected")
else:
    print("✅ Area preservation within expected limits")
print("=" * 50)

# 2) Vegetation analysis -------------------------------------------------------------------
result = get_VEGETATION(
    polygons=GRP_no_split,
    veg_raster=r".../1232-T2-TM2_1-GIS-Remote-Sensing/06_GIS-Data/09_LCC/LCC_2022_VC_CC/LCC_2022_4_1to1_CC_EPSG32638.tif",
    output_path=r"D:/KPI/vector/CC22test_ALLOPTIoptimum.gpkg",
    id_field="pid",
    by_row=True,           
    return_result=True
)

print("Success!")
print(f"Output features: {len(result)}")
print(result[['pid', 'VegArea', 'VegRatio']].head())
# FINITO -----------------------------------------------------------------------------------