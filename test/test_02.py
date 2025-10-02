from vCcCpy.core import get_VCratio, get_VCarea, crs_check_transform
import geopandas as gpd

# Vector data paths
GRP = "D:/KPI/vector/GRPtypo.geojson"
MET = "D:/KPI/vector/METROu.geojson"

# Raster data path  
R = "D:/KPI/raster/202502and08m.tif"

# CRS check, transformation and feature repair
metro_gdf = crs_check_transform(MET, R)
grp_gdf = crs_check_transform(GRP, R)

# 1) METRO VCratio -------------------------------------------------------------------------
print("Processing METRO VCratio...")
get_VCratio(
    input_raster=R,
    input_shape=metro_gdf,  
    output_shape="D:/KPI/vector/pytest/METROu25.geojson",
    id_field=None
)

# 2) GRP typologies VCratio ----------------------------------------------------------------
print("Processing GRP VCratio...")
get_VCratio(
    input_raster=R,
    input_shape=grp_gdf,  
    output_shape="D:/KPI/vector/pytest/GRPtypo25.geojson",
    id_field="MCAT"
)

# 3) METRO VCarea --------------------------------------------------------------------------
print("Processing METRO VCarea...")
get_VCarea(
    input_raster=R,
    input_shape=metro_gdf, 
    output_shape="D:/KPI/vector/pytest/METROa25.geojson", 
    id_field=None
)

print("VCratio and VCarea calculations completed!")
# FINITO -----------------------------------------------------------------------------------
