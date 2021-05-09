
import geopandas as gpd, pandas as pd
import contextily as ctx
from pyproj import CRS


df = gpd.read_file('data/geospatial/municipalities.geojson')

df['geometry'][0].plot()





