import geopandas as gpd
import sys
import os
from vCcCpy.core import get_VEGETATION, explode_pid
import time
import pandas as pd  

# 1) Data preparation ----------------------------------------------------------------------
# Load data, explode preserving GeoDataFrame structure (creating a new attribute named 'pid' concatenating the attribute defined in field_to_string)
GRP = explode_pid("D:/KPI/vector/GRPfullNAME.geojson", field_to_string="NAME_ENGLI")  

# Add 'PolyArea' column 
GRP['PolyArea'] = GRP.geometry.area

# Rename data (no splitting -> preserving all features)
GRP_no_split = GRP

# 2) Vegetation analysis -------------------------------------------------------------------
result = get_VEGETATION(
    polygons=GRP_no_split,
    veg_raster=r".../1232-T2-TM2_1-GIS-Remote-Sensing/06_GIS-Data/09_LCC/LCC_2022_VC_CC/LCC_2022_4_1to1_CC_EPSG32638.tif",
    output_path=r"D:/KPI/vector/CC22test_ALLOPTIoptimum.gpkg",
    id_field="pid",
    by_row=True,           
    return_result=True)

print(f"Output features: {len(result)}")
print(result[['pid', 'VegArea', 'VegRatio']].head())
# FINITO -----------------------------------------------------------------------------------