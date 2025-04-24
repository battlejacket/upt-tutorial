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

    def generate_grid_points(self, Lo, Ho):
        """
        Generate a grid-based point cloud and compute the signed distance function (SDF).
        """
        aspect = (self.xMax - self.xMin) / 1
        Ny = int(np.sqrt(self.totalPoints / aspect))
        Nx = int(self.totalPoints / Ny)

        x = np.linspace(self.xMin, self.xMax, Nx)
        y = np.linspace(-0.5, 0.5, Ny)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)

        geo = self.ffsGeo(Lo=Lo, Ho=Ho)
        sdf = np.array([self.signed_distance(p, geo) for p in points])

        mask = (sdf >= 0)
        insidePoints = points[mask]
        insideSdf = sdf[mask]

        return insidePoints, insideSdf

    def generate_mesh_points(self, Lo, Ho, meshParameters):
        """
        Generate a mesh-based point cloud using pygmsh and compute the signed distance function (SDF).
        """
        
        
        geo = self.ffsGeo(Lo=Lo, Ho=Ho)
        # Convert to list of [x, y]
        bPoints = [list(p.coords)[0] for p in geo['boundaryPoints']]

        with pygmsh.geo.Geometry() as geom:
            poly = geom.add_polygon(
                bPoints[:-1], 
                mesh_size=meshParameters['size'],
            )

            field0 = geom.add_boundary_layer(
                edges_list=poly.curves,
                lcmin=meshParameters['lcmin'],  # Min cell size
                lcmax=meshParameters['lcmax'],  # Max cell size
                distmin=meshParameters['distmin'],  # Min distance
                distmax=meshParameters['distmax'],  # Max distance  
            )

            geom.set_background_mesh([field0], operator="Min")
            mesh = geom.generate_mesh()
            points = mesh.points[:, :2]  # Get only x and y coordinates
        
        points = points[(points[:, 0] >= self.xMin) & (points[:, 0] <= self.xMax)]
        sdf = np.array([self.signed_distance(p, geo) for p in points])

        return points, sdf
    
    def update_base_mesh(self, Lo, Ho, meshParameters):
        """
        Generate a mesh-based point cloud by updating a mesh from training data and compute the signed distance function (SDF).
        """
        geo = self.ffsGeo(Lo=Lo, Ho=Ho)

        # ToDo: Load mesh without the obstacle

        Wo = 0.1
        thichness = 0.1 
        usx = -Lo - thichness
        dsx = -Lo + Wo + thichness
        y = 0.5 - Ho - thichness


        bPoints = [
            [usx, 0.5], [-Lo, 0.5], [-Lo, 0.5 - Ho],
            [-Lo + Wo, 0.5 - Ho], [-Lo + Wo, 0.5], [dsx, 0.5],
            [dsx, y], [usx, y]
        ]


        with pygmsh.geo.Geometry() as geom:
            poly = geom.add_polygon(
                bPoints, 
                mesh_size=meshParameters['size'],
            )

            field0 = geom.add_boundary_layer(
                edges_list=poly.curves[1:4],
                lcmin=meshParameters['lcmin'],  # Min cell size
                lcmax=meshParameters['lcmax'],  # Max cell size
                distmin=meshParameters['distmin'],  # Min distance
                distmax=meshParameters['distmax'],  # Max distance  
            )

            geom.set_background_mesh([field0], operator="Min")
            mesh = geom.generate_mesh()
            points = mesh.points[:, :2]  # Get only x and y coordinates

            # ToDo: Create a mask to remove points surrounding the obstacle (within thickness) from the original mesh

            # ToDo: Replace points surrounding the obstacle with the points from the new mesh
        
        points = points[(points[:, 0] >= self.xMin) & (points[:, 0] <= self.xMax)]
        sdf = np.array([self.signed_distance(p, geo) for p in points])

        return points, sdf

    def preprocess(self, re_value, Lo, Ho, idx, useMesh=False, meshParameters=None):
        """
        Preprocess the input data by generating the point cloud and normalizing features.
        """
        if useMesh:
            points, sdf = self.update_base_mesh(
                Lo=Lo,
                Ho=Ho,
                meshParameters=meshParameters
            )
        else:
            points, sdf = self.generate_grid_points(
                Lo=Lo,
                Ho=Ho
            )

        input_feat = self.train_dataset.normalize_sdf(torch.tensor(sdf, dtype=torch.float32))
        input_pos = self.train_dataset.normalize_pos(torch.tensor(points, dtype=torch.float32))
        re = self.train_dataset.normalize_re(torch.tensor([re_value]))
        supernode_idxs = torch.randperm(len(input_feat))[:self.numSupernodes]
        batch_idx = torch.zeros(len(input_feat), dtype=torch.int32)

        # input_feat = torch.tensor(sdf, dtype=torch.float32)
        # input_pos = torch.tensor(points, dtype=torch.float32)
        # re = torch.tensor([re_value])
        # supernode_idxs = torch.randperm(len(input_feat))[:self.numSupernodes]
        # batch_idx = torch.zeros(len(input_feat), dtype=torch.int32)

        return {
            'input_feat': input_feat.unsqueeze(1),
            'input_pos': input_pos,
            'supernode_idxs': supernode_idxs,
            'batch_idx': batch_idx,
            'output_pos': input_pos.unsqueeze(0),
            're': re
        }

    def infer(self, parameter_sets, output_pos=None, useMesh=False, meshParameters=None):
        """
        Perform inference for a set of parameters using the model.
        """
        if self.model is None or self.device is None:
            raise ValueError("Model and device must be assigned before calling infer.")
        if useMesh and meshParameters is None:
            raise ValueError("Mesh parameters must be provided when useMesh is True.")
        
        results = []
        idx = 0
        for re_value, Lo, Ho in parameter_sets:
            batch = self.preprocess(re_value, Lo, Ho, idx, useMesh=useMesh, meshParameters=meshParameters)

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

    def get_batches(self, parameter_sets, useMesh=False, meshParameters=None):
        """
        Generate and return a list of batches (size 1) for the given parameter sets.
        This method does not require a model or device to be assigned.
        """
        if useMesh and meshParameters is None:
            raise ValueError("Mesh parameters must be provided when useMesh is True.")
        
        batches = []
        idx = 0
        for re_value, Lo, Ho in parameter_sets:
            batch = self.preprocess(re_value, Lo, Ho, idx, useMesh=useMesh, meshParameters=meshParameters)
            batches.append(batch)
            idx += 1
        return batches