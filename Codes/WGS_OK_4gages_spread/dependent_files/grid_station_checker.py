import pandas as pd, json

f = "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/dependent_files/grid_grouped_candidates_wgs84.csv"
df = pd.read_csv(f)
x=2201
row = df[df["id"] == x].iloc[0]

for col in ["quadrant_groups", "sector3_groups"]:
    print(f"\n--- {col} ---")
    groups = json.loads(row[col]) if pd.notna(row[col]) and str(row[col]).strip() else []
    for g in groups:
        print(g["ids"])
print("for station ",x)