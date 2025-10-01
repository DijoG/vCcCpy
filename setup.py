from setuptools import setup

setup(
    name="vCcCpy",
    version="0.1.0",
    packages=["vCcCpy"],  
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.0", 
        "geopandas>=0.14.0",
        "rasterio>=1.3.8",
        "rioxarray>=0.15.0",
        "shapely>=2.0.2",
        "pyproj>=3.6.1",
        "tqdm>=4.66.0",
        "dask>=2023.10.0",
        "dask-geopandas>=0.3.0",
    ],
    python_requires=">=3.8",
)