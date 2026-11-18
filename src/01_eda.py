import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
import folium

DATA_DIR = "data/Student folder"
OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

def inspect_partition(partition_name):
    part_path = os.path.join(DATA_DIR, partition_name)
    trip_dirs = sorted([d for d in glob.glob(os.path.join(part_path, "*")) if os.path.isdir(d)])
    records = []

    for tdir in trip_dirs:
        trip_id = os.path.basename(tdir)
        track_f = os.path.join(tdir, f"{trip_id}.track")
        nodes_f = os.path.join(tdir, f"{trip_id}.nodes")
        arcs_f = os.path.join(tdir, f"{trip_id}.arcs")
        route_f = os.path.join(tdir, f"{trip_id}.route")

        track = pd.read_csv(track_f, sep=r"\s+", header=None, names=["lon", "lat", "ts"])
        nodes = pd.read_csv(nodes_f, sep=r"\s+", header=None, names=["lon", "lat"])
        arcs = pd.read_csv(arcs_f, sep=r"\s+", header=None, names=["u", "v"])
        route = pd.read_csv(route_f, sep=r"\s+", header=None, names=["arc_id"]) if os.path.exists(route_f) else None

        dt = track["ts"].diff().dropna()
        G = nx.DiGraph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from(zip(arcs["u"], arcs["v"]))

        is_route_connected = True
        if route is not None and len(route) > 1:
            r_arcs = arcs.iloc[route["arc_id"].values]
            is_route_connected = all(r_arcs["v"].values[:-1] == r_arcs["u"].values[1:])

        records.append({
            "partition": partition_name,
            "trip_id": trip_id,
            "n_gps": len(track),
            "n_nodes": len(nodes),
            "n_arcs": len(arcs),
            "n_route_arcs": len(route) if route is not None else 0,
            "route_connected": is_route_connected,
            "lon_min": track["lon"].min(),
            "lon_max": track["lon"].max(),
            "lat_min": track["lat"].min(),
            "lat_max": track["lat"].max(),
            "missing_gps": track.isna().sum().sum(),
            "dt_1s_pct": (dt == 1.0).mean() * 100 if len(dt) > 0 else 0,
            "dt_max": dt.max() if len(dt) > 0 else 0,
            "isolated_nodes": nx.number_of_isolates(G),
            "weak_components": nx.number_weakly_connected_components(G),
        })

    return pd.DataFrame(records)

df_train = inspect_partition("train")
df_test = inspect_partition("student_test")
df_all = pd.concat([df_train, df_test], ignore_index=True)

print("=== PARTITION SUMMARY ===")
print(df_all.groupby("partition")[["n_gps", "n_nodes", "n_arcs", "n_route_arcs", "dt_1s_pct", "dt_max", "weak_components"]].describe().round(2))
print("\nCoordinate Bounding Box (Overall):")
print(f"Lon: [{df_all['lon_min'].min():.5f}, {df_all['lon_max'].max():.5f}]")
print(f"Lat: [{df_all['lat_min'].min():.5f}, {df_all['lat_max'].max():.5f}]")
print(f"Trips with disconnected reference route sequences: {(~df_all['route_connected']).sum()} / {len(df_all)}")

df_all.to_csv("artifacts/eda_inventory.csv", index=False)

sample_id = "00000001"
s_dir = os.path.join(DATA_DIR, "train", sample_id)
track = pd.read_csv(os.path.join(s_dir, f"{sample_id}.track"), sep=r"\s+", header=None, names=["lon", "lat", "ts"])
nodes = pd.read_csv(os.path.join(s_dir, f"{sample_id}.nodes"), sep=r"\s+", header=None, names=["lon", "lat"])
arcs = pd.read_csv(os.path.join(s_dir, f"{sample_id}.arcs"), sep=r"\s+", header=None, names=["u", "v"])
route = pd.read_csv(os.path.join(s_dir, f"{sample_id}.route"), sep=r"\s+", header=None, names=["arc_id"])

m = folium.Map(location=[track["lat"].mean(), track["lon"].mean()], zoom_start=15)

gt_group = folium.FeatureGroup(name="Reference Route (.route)")
for aid in route["arc_id"]:
    u, v = arcs.iloc[aid]
    p1 = [nodes.iloc[u]["lat"], nodes.iloc[u]["lon"]]
    p2 = [nodes.iloc[v]["lat"], nodes.iloc[v]["lon"]]
    folium.PolyLine([p1, p2], color="green", weight=5, opacity=0.8).add_to(gt_group)
gt_group.add_to(m)

gps_group = folium.FeatureGroup(name="Raw GPS (.track)")
for _, row in track.iterrows():
    folium.CircleMarker([row["lat"], row["lon"]], radius=3, color="blue", fill=True, fill_opacity=0.7).add_to(gps_group)
gps_group.add_to(m)

folium.LayerControl().add_to(m)
m.save(os.path.join(OUT_DIR, "eda_trip_00000001.html"))
print(f"\nSaved Folium map to {OUT_DIR}/eda_trip_00000001.html")
