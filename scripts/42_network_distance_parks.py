"""
Network Distance to Park Boundaries (NDVI Polygons)
Economic Impact Report for Butler County Parks

This script computes road-network distance from each residential parcel to
the nearest park boundary using the county roads layer. It produces two
distance measures:
  - All NDVI parks (OSM-derived polygons)
  - Butler County MetroParks only (by operator and name match)
"""

import json
import math
import os
import heapq

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import shape, Point
from shapely.ops import nearest_points
from shapely.strtree import STRtree


def load_parks_from_ndvi(csv_path, target_crs):
    df = pd.read_csv(csv_path)
    df = df[df[".geo"].notna()].copy()
    df["geometry"] = df[".geo"].apply(lambda g: shape(json.loads(g)))
    gdf = gpd.GeoDataFrame(df.drop(columns=[".geo"]), geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs(target_crs)

    if "leisure" in gdf.columns or "landuse" in gdf.columns:
        leisure = gdf["leisure"].fillna("")
        landuse = gdf["landuse"].fillna("")
        gdf = gdf[(leisure == "park") | (landuse == "park")].copy()

    return gdf


def filter_residential(parcels_gdf):
    land_use_col = None
    for col in ["CLASS", "LANDUSE", "LUC"]:
        if col in parcels_gdf.columns:
            land_use_col = col
            break
    if land_use_col is None:
        return parcels_gdf.copy()
    return parcels_gdf[parcels_gdf[land_use_col] == "R"].copy()


def build_road_graph(roads_gdf):
    graph = nx.Graph()
    node_map = {}
    lines = []
    start_nodes = []
    end_nodes = []

    def get_node_id(coord):
        node_id = node_map.get(coord)
        if node_id is None:
            node_id = len(node_map)
            node_map[coord] = node_id
        return node_id

    for geom in roads_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        geoms = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for line in geoms:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            start = coords[0]
            end = coords[-1]
            u = get_node_id(start)
            v = get_node_id(end)
            length = float(line.length)
            graph.add_edge(u, v, weight=length)
            lines.append(line)
            start_nodes.append(u)
            end_nodes.append(v)

    tree = STRtree(lines)
    line_index = {id(lines[i]): i for i in range(len(lines))}

    return graph, lines, start_nodes, end_nodes, tree, line_index


def nearest_line_index(tree, line_index, geom):
    nearest = tree.nearest(geom)
    if isinstance(nearest, (int, np.integer)):
        return int(nearest)
    return line_index[id(nearest)]


def seed_distances_from_parks(parks_gdf, lines, start_nodes, end_nodes, tree, line_index):
    initial = {}
    for geom in parks_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        boundary = geom.boundary if geom.geom_type != "Point" else geom
        idx = nearest_line_index(tree, line_index, boundary)
        line = lines[idx]

        pt_line, pt_boundary = nearest_points(line, boundary)
        off_network = pt_line.distance(pt_boundary)
        proj = line.project(pt_line)
        dist_to_start = proj + off_network
        dist_to_end = (line.length - proj) + off_network

        u = start_nodes[idx]
        v = end_nodes[idx]
        if dist_to_start < initial.get(u, math.inf):
            initial[u] = dist_to_start
        if dist_to_end < initial.get(v, math.inf):
            initial[v] = dist_to_end

    return initial


def multi_source_dijkstra(graph, initial):
    dist = {node: math.inf for node in graph.nodes}
    heap = []
    for node, d in initial.items():
        dist[node] = d
        heapq.heappush(heap, (d, node))

    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v, data in graph[u].items():
            nd = d + data["weight"]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist


def compute_parcel_network_distances(parcels_gdf, lines, start_nodes, end_nodes, tree, line_index, node_dist):
    distances = []
    total = len(parcels_gdf)
    for idx, geom in enumerate(parcels_gdf.geometry):
        if geom is None or geom.is_empty:
            distances.append(math.inf)
            continue
        point = geom if geom.geom_type == "Point" else geom.centroid
        line_idx = nearest_line_index(tree, line_index, point)
        line = lines[line_idx]

        pt_line, pt_point = nearest_points(line, point)
        off_network = pt_line.distance(pt_point)
        proj = line.project(pt_line)

        u = start_nodes[line_idx]
        v = end_nodes[line_idx]
        du = node_dist.get(u, math.inf)
        dv = node_dist.get(v, math.inf)
        best = min(proj + du, (line.length - proj) + dv)
        distances.append(off_network + best)

        if (idx + 1) % 10000 == 0 or (idx + 1) == total:
            print(f"  Processed {idx + 1:,} / {total:,} parcels")

    return distances


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parks_path = os.path.join(project_root, "data_raw", "ButlerParks_NDVI_mean_S2_2025_JunAug.csv")
    roads_path = os.path.join(project_root, "data_raw", "Roads", "Roads.shp")
    parcels_path = os.path.join(project_root, "data_raw", "CURRENTPARCELS", "CURRENTPARCELS.shp")
    output_path = os.path.join(project_root, "data_intermediate", "parcel_network_distances.csv")

    print("Loading roads...")
    roads = gpd.read_file(roads_path)
    target_crs = roads.crs

    print("Loading parcels...")
    parcels = gpd.read_file(parcels_path)
    if parcels.crs != target_crs:
        parcels = parcels.to_crs(target_crs)
    parcels = filter_residential(parcels)

    if "PIN" not in parcels.columns:
        raise ValueError("PIN column not found in parcels layer.")

    print(f"Residential parcels: {len(parcels):,}")

    print("Loading NDVI park boundaries...")
    parks_all = load_parks_from_ndvi(parks_path, target_crs)
    print(f"All parks loaded: {len(parks_all):,}")

    metropark_names = {
        "Rentschler Forest",
        "Voice of America",
        "Gilmore MetroPark",
        "Forest Run",
        "Indian Creek",
        "Governor Bebb",
        "Sebald / Elk Creek",
        "Chrisholm Historic Farmstead",
        "Dudley Woods",
        "Bicentennial Commons",
        "Meadow Ridge",
        "Four Mile Creek",
        "Woodsdale MetroPark",
    }

    operator = parks_all["operator"].fillna("").str.lower()
    name = parks_all["name"].fillna("")
    metro_mask = operator.str.contains("metroparks of butler county", na=False) | name.isin(metropark_names)
    parks_metro = parks_all[metro_mask].copy()
    print(f"Metroparks subset: {len(parks_metro):,}")

    print("Building road network...")
    graph, lines, start_nodes, end_nodes, tree, line_index = build_road_graph(roads)
    print(f"Road graph nodes: {graph.number_of_nodes():,}, edges: {graph.number_of_edges():,}")

    print("Seeding distances for ALL parks...")
    init_all = seed_distances_from_parks(parks_all, lines, start_nodes, end_nodes, tree, line_index)
    dist_all = multi_source_dijkstra(graph, init_all)

    print("Seeding distances for METROPARKS...")
    init_metro = seed_distances_from_parks(parks_metro, lines, start_nodes, end_nodes, tree, line_index)
    dist_metro = multi_source_dijkstra(graph, init_metro)

    print("Computing parcel network distances (ALL parks)...")
    dist_all_parcels = compute_parcel_network_distances(
        parcels, lines, start_nodes, end_nodes, tree, line_index, dist_all
    )

    print("Computing parcel network distances (METROPARKS)...")
    dist_metro_parcels = compute_parcel_network_distances(
        parcels, lines, start_nodes, end_nodes, tree, line_index, dist_metro
    )

    # Convert feet to miles
    dist_all_miles = [d / 5280.0 if math.isfinite(d) else math.nan for d in dist_all_parcels]
    dist_metro_miles = [d / 5280.0 if math.isfinite(d) else math.nan for d in dist_metro_parcels]

    out_df = pd.DataFrame({
        "PIN": parcels["PIN"].values,
        "dist_net_miles_all": dist_all_miles,
        "dist_net_miles_metro": dist_metro_miles,
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
