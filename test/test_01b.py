from vCcCpy.core import aggregate_by_field

# Aggregation ------------------------------------------------------------------------------
if __name__ == "__main__":
    input_gpkg = r"D:/KPI/vector/CC22test_LPoptimum.gpkg"
    output_gpkg = r"D:/KPI/vector/CC22test_LPoptimum_aggregated.gpkg"
    
    result = aggregate_by_field(input_gpkg, output_gpkg, group_by='pid')
    print(result[['pid', 'MCAT', 'PolyArea', 'VegArea', 'VCratio']].head())
# FINITO -----------------------------------------------------------------------------------