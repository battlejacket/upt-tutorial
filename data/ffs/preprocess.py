# conda create --name open3d python=3.9
# pip install open3d
# pip install meshio
# pip install torch
# pip install tempfile
import os
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from csv_rw import csv_to_dict
import networkx as nx
from shapely.geometry import Point, LineString, Polygon


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="e.g. /data/shapenet_car/training_data")
    parser.add_argument("--dst", type=str, required=True, help="e.g. /data/shapenet_car/preprocessed")
    parser.add_argument("--saveNorm", type=bool, required=False, help="Set to False to avoid overwriting the normalization parameters")
    parser.add_argument("--sampleRatio", type=bool, required=False, help="ratio of points to include")
    parser.add_argument("--sampleSeed", type=bool, required=False, help="Seed for random sampling")
    return vars(parser.parse_args())


# def order_points_tsp(points):
#     G = nx.complete_graph(len(points))
#     for i in range(len(points)):
#         for j in range(i + 1, len(points)):
#             dist = np.linalg.norm(points[i] - points[j])
#             G[i][j]["weight"] = dist
#             G[j][i]["weight"] = dist
#     tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=True)
#     ordered_points = points[tsp_path]
#     return ordered_points

# def signed_distance(p, shape):
#     point = Point(p)
#     distance = point.distance(shape)
#     # Negative if inside the shape
#     if shape.contains(point):
#         return -distance
#     else:
#         return distance

# def compute_sdf(mesh_points, boundary_points):
#     # Compute signed distance function (SDF) for mesh points
#     # Order boundary points using TSP
#     ordered = order_points_tsp(boundary_points)
#     # Define outline (boundary) shape
#     boundary = LineString(ordered)
#     # Compute signed distances
#     sdf_values = np.array([signed_distance(p, boundary) for p in mesh_points])
#     return sdf_values

def ffsGeo(Ho, Lo):
    xMax = 12
    xMin = -6
    Wo = 0.1
    bPoints = [
        Point(xMin, 0.5), Point(-Lo, 0.5), Point(-Lo, 0.5 - Ho),
        Point(-Lo + Wo, 0.5 - Ho), Point(-Lo + Wo, 0.5), Point(0, 0.5),
        Point(0, 0), Point(xMax, 0), Point(xMax, -0.5), Point(xMin, -0.5),
        Point(xMin, 0.5)
    ]
    geo = Polygon(bPoints)
    boundary = LineString(bPoints)
    return {'boundary': boundary, 'geo': geo}

def signed_distance(p, shape):
    pt = Point(p)
    d = pt.distance(shape['boundary'])
    return d if shape['geo'].contains(pt) else -d



def main(src, dst, compute_sdf_values = True, save_normalization_param = True):
    src = Path(src).expanduser()
    assert src.exists(), f"'{src.as_posix()}' doesnt exist"
    # assert src.name == "training_data"
    dst = Path(dst).expanduser()
    if not dst.exists():
        os.mkdir(dst)
    # assert not dst.exists(), f"'{dst.as_posix()}' exist"
    print(f"src: {src.as_posix()}")
    print(f"dst: {dst.as_posix()}")
    
    
    # find all uris for samples
    uris = []
    for root, dirs, files in os.walk(src):
        for file in files:
            filePath = Path(os.path.join(root, file))
            uris.append(filePath)
    uris.sort()
    print(f"found {len(uris)} samples")
        
    # define csv parameters
    csvInvarNames = ["Points:0", "Points:1"]
    dictInvarNames = ["x", "y"]
    csvOutvarNames = ["Velocity:0", "Velocity:1", "Pressure"]
    dictOutvarNames = ["u", "v", "p"]
    scales = {"p": (0,1), "u": (0,1), "v": (0,1), "x": (0,1), "y": (-0.5,1)} #(translation, scale)
    skiprows = 0
    csvVarNames = csvInvarNames + csvOutvarNames
    dictVarNames = dictInvarNames + dictOutvarNames

    mapping = {}
    for csvVarName, dictVarName in zip(csvVarNames, dictVarNames):
        mapping[csvVarName] = dictVarName
    
    # Initialize variables for min/max coordinates and mean/std calculations
    min_coords = torch.tensor([float('inf'), float('inf')])
    max_coords = torch.tensor([-float('inf'), -float('inf')])
    sum_vars = torch.tensor([0.0, 0.0, 0.0, 0.0])  # For u, v, p, sdf
    sum_sq_vars = torch.tensor([0.0, 0.0, 0.0, 0.0])  # For u^2, v^2, p^2, sdf^2
    total_samples = 0

        
    for uri in tqdm(uris):
        reluri = uri.relative_to(src).with_suffix('')
        out = dst / reluri
        out.mkdir(exist_ok=True, parents=True)
        
        #read and process csv
        csvData = csv_to_dict(uri, mapping=mapping, delimiter=",", skiprows=skiprows)
        for key in dictVarNames:
            csvData[key] += scales[key][0]
            csvData[key] /= scales[key][1]
        
        # Save mesh points
        NpMesh=np.concat([csvData["x"], csvData["y"]], axis=1)
        mesh_points = torch.from_numpy(NpMesh).float()
        min_coords = torch.min(min_coords, mesh_points.min(dim=0).values)
        max_coords = torch.max(max_coords, mesh_points.max(dim=0).values)
        torch.save(mesh_points, out / "mesh_points.th")

        # Save target variables
        for i, outVar in enumerate(dictOutvarNames):  # u, v, p
            data = torch.tensor(csvData[outVar]).float()
            torch.save(data, out / f"{outVar}.th")
            sum_vars[i] += data.sum()
            sum_sq_vars[i] += (data ** 2).sum()
        total_samples += data.numel()

        if compute_sdf_values:

            parameters = str(uri).split('DP')[1].replace('.csv', '').replace(',', '.').split('_')[1:]

            # re = float(parameters[0])
            Lo = float(parameters[1])
            Ho = float(parameters[2])

            # Compute and store baseline SDF
            geo = ffsGeo(Ho=Ho, Lo=Lo)
            sdf_values = torch.tensor([signed_distance(p, geo) for p in mesh_points]).float()

            # sdf
            # Read boundary points csv
            # boundaryCsvFile = str(uri).split('_')[0].replace('CSV600', 'boundary600') + '_bound.csv'
            # print(f"boundaryCsvFile: {boundaryCsvFile}")
            # boundaryCsvDict = csv_to_dict(boundaryCsvFile, mapping=mapping, delimiter=",", skiprows=skiprows)
            # for key in dictVarNames:
            #     boundaryCsvDict[key] += scales[key][0]
            #     boundaryCsvDict[key] /= scales[key][1]
            # boundaryPoints = np.concat([boundaryCsvDict["x"], boundaryCsvDict["y"]], axis=1)

            # # Save sdf3
            # print('computing sdf')
            # sdf_values = torch.tensor(compute_sdf(mesh_points, boundaryPoints)).float()
            # print('sdf computed')
            torch.save(sdf_values, out / "mesh_sdf.th")
            sum_vars[-1] += sdf_values.sum()
            sum_sq_vars[-1] += (sdf_values ** 2).sum()
            # total_samples += sdf_values.numel()

    if save_normalization_param:
        # Calculate mean and std
        mean_vars = sum_vars / total_samples
        std_vars = torch.sqrt((sum_sq_vars / total_samples) - (mean_vars ** 2))

        # Save normalization parameters
        torch.save({"min_coords": min_coords, "max_coords": max_coords}, dst / "coords_norm.th")
        torch.save({"mean": mean_vars, "std": std_vars}, dst / "vars_norm.th")

    print("fin")


if __name__ == "__main__":
    # main(**parse_args())
    main('./data/ffs/CSV600/', './data/ffs/preprocessed600/', compute_sdf_values=True, save_normalization_param=True)
