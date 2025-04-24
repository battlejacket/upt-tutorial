from shapely.geometry import Point, LineString, Polygon
import numpy as np
import torch
import pygmsh

class ffsInference:
    def __init__(self, train_dataset, totalPoints, numSupernodes, model = None, device = None, xMin = None, xMax = None):
        self.model = model
        self.train_dataset = train_dataset
        self.totalPoints = totalPoints
        self.numSupernodes = numSupernodes
        self.device = device
        if xMin == None:
            self.xMin = train_dataset.crop_values[0][0]
        else: 
            self.xMin = xMin
        if xMax == None:
            self.xMax = train_dataset.crop_values[1][0]
        else: 
            self.xMax = xMax

    def ffsGeo(self, Lo, Ho):
        xMax = 12
        xMin = -6
        Wo = 0.1
        
        bPoints = [
            Point(xMin, 0.5), Point(-Lo, 0.5), Point(-Lo, 0.5 - Ho),
            Point(-Lo + Wo, 0.5 - Ho), Point(-Lo + Wo, 0.5), Point(0, 0.5),
            Point(0, 0), Point(xMax, 0), Point(xMax, -0.5), Point(xMin, -0.5),
            Point(xMin, 0.5)
        ]
        boundary = LineString(bPoints)
        geo = Polygon(bPoints)

        return {'boundary': boundary, 'boundaryPoints': bPoints, 'geo': geo}

    def signed_distance(self, p, shape):
        pt = Point(p)
        d = pt.distance(shape['boundary'])
        return d if shape['geo'].contains(pt) else -d

    def generate_grid_points(self, xMin, xMax, Lo, Ho):
        """
        Generate a grid-based point cloud and compute the signed distance function (SDF).
        """
        aspect = (xMax - xMin) / 1
        Ny = int(np.sqrt(self.totalPoints / aspect))
        Nx = int(self.totalPoints / Ny)

        x = np.linspace(xMin, xMax, Nx)
        y = np.linspace(-0.5, 0.5, Ny)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)

        geo = self.ffsGeo(Lo=Lo, Ho=Ho)
        sdf = np.array([self.signed_distance(p, geo) for p in points])

        mask = (sdf >= 0)
        insidePoints = points[mask]
        insideSdf = sdf[mask]

        return insidePoints, insideSdf

    def generate_mesh_points(self, Lo, Ho):
        """
        Generate a mesh-based point cloud using pygmsh and compute the signed distance function (SDF).
        """
        geo = self.ffsGeo(Lo=Lo, Ho=Ho)
        bPoints = geo['boundaryPoints']  # Reuse boundary points from ffsGeo

        with pygmsh.geo.Geometry() as geom:
            poly = geom.add_polygon(
                bPoints[:-1],  # Exclude the last point to avoid duplication
                mesh_size=0.3,
            )

            field0 = geom.add_boundary_layer(
                edges_list=poly.curves[1:4],
                lcmin=0.05,  # Min cell size
                lcmax=0.2,  # Max cell size
                distmin=0.0,  # Min wall distance
                distmax=0.2,  # Max wall distance
            )

            geom.set_background_mesh([field0], operator="Min")
            mesh = geom.generate_mesh()
            points = mesh.points

        sdf = np.array([self.signed_distance(p, geo) for p in points])

        mask = (sdf >= 0)
        insidePoints = points[mask]
        insideSdf = sdf[mask]

        return insidePoints, insideSdf

    def generate_point_cloud(self, xMin, xMax, Lo, Ho, useMesh=False):
        """
        Generate a point cloud (grid-based or mesh-based) and compute the signed distance function (SDF).
        """
        if useMesh:
            return self.generate_mesh_points(Lo, Ho)
        else:
            return self.generate_grid_points(xMin, xMax, Lo, Ho)

    def preprocess(self, re_value, Lo, Ho, idx, useMesh=False):
        """
        Preprocess the input data by generating the point cloud and normalizing features.
        """
        insidePoints, insideSdf = self.generate_point_cloud(
            xMin=self.xMin,
            xMax=self.xMax,
            Lo=Lo,
            Ho=Ho,
            useMesh=useMesh
        )

        input_feat = self.train_dataset.normalize_sdf(torch.tensor(insideSdf, dtype=torch.float32))
        input_pos = self.train_dataset.normalize_pos(torch.tensor(insidePoints, dtype=torch.float32))
        re = self.train_dataset.normalize_re(torch.tensor([re_value]))
        supernode_idxs = torch.randperm(len(input_feat))[:self.numSupernodes]
        batch_idx = torch.zeros(len(input_feat), dtype=torch.int32)

        return {
            'input_feat': input_feat.unsqueeze(1),
            'input_pos': input_pos,
            'supernode_idxs': supernode_idxs,
            'batch_idx': batch_idx,
            'output_pos': input_pos.unsqueeze(0),
            're': re
        }

    def infer(self, parameter_sets, output_pos=None, useMesh=False):
        """
        Perform inference for a set of parameters using the model.
        """
        if self.model is None or self.device is None:
            raise ValueError("Model and device must be assigned before calling infer.")

        results = []
        idx = 0
        for re_value, Lo, Ho in parameter_sets:
            batch = self.preprocess(re_value, Lo, Ho, idx, useMesh=useMesh)

            current_output_pos = output_pos if output_pos is not None else batch['output_pos']

            with torch.no_grad():
                y_hat = self.model(
                    input_feat=batch['input_feat'].to(self.device),
                    input_pos=batch['input_pos'].to(self.device),
                    supernode_idxs=batch['supernode_idxs'].to(self.device),
                    batch_idx=batch['batch_idx'].to(self.device),
                    output_pos=current_output_pos.to(self.device),
                    re=batch['re'].to(self.device),
                )
                result = dict(
                    parameters={'re': re_value, 'Lo': Lo, 'Ho': Ho},
                    points=self.train_dataset.denormalize_pos(current_output_pos.squeeze()),
                    prediction=self.train_dataset.denormalize_feat(y_hat.cpu())
                )
                results.append(result)
                idx += 1
        return results

    def get_batches(self, parameter_sets, useMesh=False):
        """
        Generate and return a list of batches for the given parameter sets.
        This method does not require a model or device to be assigned.
        """
        batches = []
        idx = 0
        for re_value, Lo, Ho in parameter_sets:
            batch = self.preprocess(re_value, Lo, Ho, idx, useMesh=useMesh)
            batches.append(batch)
            idx += 1
        return batches