import pandas as pd
import numpy as np

# ---- radar params (from one .out file) ----
xll = -96.920830
yll = 36.737499
cell = 0.004167
nrows = 995

lon0 = xll + 0.5 * cell
lat_top = yll + (nrows - 0.5) * cell

# ---- load your OK grid ----
grid = pd.read_csv("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/dependent_files/grid_centers_wgs84.csv")

# take a subset
sample = grid.sample(50, random_state=0)

rows = (lat_top - sample["Latitude"]) / cell
cols = (sample["Longitude"] - lon0) / cell

# fractional parts
row_frac = np.abs(rows - np.round(rows))
col_frac = np.abs(cols - np.round(cols))

print("Mean row frac:", row_frac.mean())
print("Mean col frac:", col_frac.mean())

print("Max row frac:", row_frac.max())
print("Max col frac:", col_frac.max())