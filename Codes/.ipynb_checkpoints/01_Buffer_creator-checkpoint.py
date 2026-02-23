import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import pyproj
from pyproj import Transformer
from shapely.geometry import Point
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

centroid_gpkg_path = "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/Selected_Selected_grids1.gpkg"
centroid_layer_name="Selected_centroids"
grid_gpkg_path="/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/Selected_Selected_grids.gpkg"
grid_layer_name="Selected_Selected_grids"
path6892513_shp="/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp"
path6893080_shp="/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp"
path6893390_shp="/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp"

points_gdf=gpd.read_file(centroid_gpkg_path)#,layer="Selected_Selected_grids1 — selected_centroids__centroids")
grids_gdf=gpd.read_file(grid_gpkg_path,layer="selected_grids__grid")
shp6892513_gdf=gpd.read_file(path6892513_shp)
shp6893080_gdf=gpd.read_file(path6893080_shp)
shp6893390_gdf=gpd.read_file(path6893390_shp)

shp6892513_gdf=shp6892513_gdf.to_crs(epsg=26915)
shp6892513_gdf['legend_label'] = '6892513'
shp6893080_gdf=shp6893080_gdf.to_crs(epsg=26915)
shp6893080_gdf['legend_label'] = '6893080'
shp6893390_gdf=shp6893390_gdf.to_crs(epsg=26915)
shp6893390_gdf['legend_label'] = '6893390'
point_gdf_26915 = points_gdf.to_crs(epsg=26915)

grids_26915  = grids_gdf.to_crs(epsg=26915).copy()
points_26915 = points_gdf.to_crs(epsg=26915).copy()

OUT_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Buffer_information")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def select_grids_and_centroids_for_catchment(
    catchment_gdf: gpd.GeoDataFrame,
    catchment_name: str,
    grids: gpd.GeoDataFrame,
    centroids: gpd.GeoDataFrame,
    out_dir: Path
):
    # dissolve catchment to one geometry (handles multipart / multiple features)
    catch_union = catchment_gdf.geometry.unary_union

    # 1) grids that intersect catchment
    grids_sel = grids[grids.intersects(catch_union)].copy()

    # 2) centroids that fall within the selected grids
    # Use spatial join for correctness + speed
    # (predicate="within": point within polygon)
    cent_sel = gpd.sjoin(
        centroids,
        grids_sel[["geometry"]].copy(),
        how="inner",
        predicate="within"
    ).drop(columns=["index_right"], errors="ignore")

    # optional: keep only unique points if duplicates occur
    cent_sel = cent_sel.drop_duplicates(subset=["geometry"])

    # Write outputs (one gpkg per catchment, two layers inside)
    out_gpkg = out_dir / f"{catchment_name}_grids_centroids.gpkg"
    grids_sel.to_file(out_gpkg, layer="selected_grids", driver="GPKG")
    cent_sel.to_file(out_gpkg, layer="selected_centroids", driver="GPKG")

    print(f"[OK] {catchment_name}: grids={len(grids_sel)}, centroids={len(cent_sel)} -> {out_gpkg}")

    return grids_sel, cent_sel

# Run for each catchment
grids_6892513, cent_6892513 = select_grids_and_centroids_for_catchment(
    shp6892513_gdf, "6892513", grids_26915, points_26915, OUT_DIR
)

grids_6893080, cent_6893080 = select_grids_and_centroids_for_catchment(
    shp6893080_gdf, "6893080", grids_26915, points_26915, OUT_DIR
)

grids_6893390, cent_6893390 = select_grids_and_centroids_for_catchment(
    shp6893390_gdf, "6893390", grids_26915, points_26915, OUT_DIR
)

buffer_km_list = [3, 4, 5, 6, 7, 8]
CENTROID_ID_COL = "ID"   # change if needed (e.g., "fid")

def count_stations_within_radius(
    centroids_gdf: gpd.GeoDataFrame,
    stations_gdf: gpd.GeoDataFrame,
    radius_m: float,
    centroid_id_col: str = "ID",
) -> pd.DataFrame:
    """
    For each centroid, count stations within a buffer radius.
    Returns a table with centroid coords + count.
    """
    centroids_gdf = centroids_gdf.to_crs(epsg=26915).copy()
    stations_gdf  = stations_gdf.to_crs(epsg=26915).copy()

    stations_sindex = stations_gdf.sindex

    rows = []
    for idx, crow in centroids_gdf.iterrows():
        cgeom = crow.geometry
        buf = cgeom.buffer(radius_m)

        # bbox candidates
        cand_idx = list(stations_sindex.intersection(buf.bounds))
        cand = stations_gdf.iloc[cand_idx]

        # exact within
        n_in = int(cand.within(buf).sum())

        cid = crow[centroid_id_col] if (centroid_id_col in centroids_gdf.columns) else idx

        rows.append({
            "centroid_id": cid,
            "centroid_x": cgeom.x,
            "centroid_y": cgeom.y,
            "buffer_m": radius_m,
            "buffer_km": radius_m / 1000.0,
            "n_stations": n_in,
        })

    return pd.DataFrame(rows)

def run_buffers_for_catchment(
    catchment_name: str,
    catchment_centroids: gpd.GeoDataFrame,
    stations_gdf: gpd.GeoDataFrame,
    buffer_km_list: list[int],
    out_dir: Path,
    centroid_id_col: str = "ID",
):
    """
    Writes one CSV per buffer distance for the given catchment.
    """
    for km in buffer_km_list:
        radius_m = km * 1000.0
        df_out = count_stations_within_radius(
            catchment_centroids,
            stations_gdf,
            radius_m,
            centroid_id_col=centroid_id_col,
        )

        out_csv = out_dir / f"{catchment_name}_stations_within_{km:02d}km.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"[OK] {catchment_name}: wrote {out_csv} (rows={len(df_out)})")

def plot_combined_histograms(catchment_name, out_dir, buffer_km_list):
    
    n_buffers = len(buffer_km_list)
    ncols = 3
    nrows = int(np.ceil(n_buffers / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for ax_idx, km in enumerate(buffer_km_list):
        ax = axes[ax_idx]
        csv_path = out_dir / f"{catchment_name}_stations_within_{km:02d}km.csv"

        if not csv_path.exists():
            ax.set_title(f"{km} km (missing)")
            ax.axis("off")
            continue

        df = pd.read_csv(csv_path)
        counts = df["n_stations"].dropna().astype(int).values

        if counts.size == 0:
            ax.set_title(f"{km} km (no data)")
            ax.axis("off")
            continue

        bins = np.arange(counts.min() - 0.5,
                         counts.max() + 1.5,
                         1)

        n, bin_edges, patches = ax.hist(
            counts,
            bins=bins,
            edgecolor="black",
            linewidth=1.2
        )

        ax.set_title(f"{km} km buffer")
        ax.set_xlabel("Stations")
        ax.set_xticks(np.arange(int(counts.min()), int(counts.max()) + 1, 1))

        # Add bar counts
        for i, p in enumerate(patches):
            height = n[i]
            if height > 0:
                x = p.get_x() + p.get_width() / 2
                ax.text(x, height, f"{int(height)}",
                        ha="center", va="bottom", fontsize=9)

    # Hide unused subplots if any
    for j in range(len(buffer_km_list), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{catchment_name} – Station count distributions by buffer radius", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.save(OUT_DIR,f"{catchment_name}.png"


# -------------------------
# Run for each catchment
# -------------------------
run_buffers_for_catchment("6892513", cent_6892513, stations, buffer_km_list, OUT_DIR, centroid_id_col=CENTROID_ID_COL)
run_buffers_for_catchment("6893080", cent_6893080, stations, buffer_km_list, OUT_DIR, centroid_id_col=CENTROID_ID_COL)
run_buffers_for_catchment("6893390", cent_6893390, stations, buffer_km_list, OUT_DIR, centroid_id_col=CENTROID_ID_COL)

# Run for each catchment
plot_combined_histograms("6892513", OUT_DIR, buffer_km_list)
plot_combined_histograms("6893080", OUT_DIR, buffer_km_list)
plot_combined_histograms("6893390", OUT_DIR, buffer_km_list)