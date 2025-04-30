from shapely.geometry import Point, LineString, Polygon
import numpy as np
import torch
import pygmsh
from data.ffs.readParameters import readParametersFromFileName
import time  # Add this import at the top of the file


class ffsInference:
    def __init__(self, train_dataset, totalPoints, numSupernodes,
                  model = None, device = None, xMin = None, xMax = None, useMesh=None, meshParameters=None):
        
        self.model = model
        self.useMesh = useMesh
        self.meshParameters = meshParameters
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
        self.obstacleMesh = None
        self.baseMesh = None





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

    def generate_mesh_points(self, Lo, Ho):
        """
        Generate a mesh-based point cloud using pygmsh and compute the signed distance function (SDF).
        """
        
        
        geo = self.ffsGeo(Lo=Lo, Ho=Ho)
        # Convert to list of [x, y]
        bPoints = [list(p.coords)[0] for p in geo['boundaryPoints']]

        with pygmsh.geo.Geometry() as geom:
            poly = geom.add_polygon(
                bPoints[:-1], 
                mesh_size=self.meshParameters['size'],
            )

            field0 = geom.add_boundary_layer(
                edges_list=poly.curves,
                lcmin=self.meshParameters['lcmin'],  # Min cell size
                lcmax=self.meshParameters['lcmax'],  # Max cell size
                distmin=self.meshParameters['distmin'],  # Min distance
                distmax=self.meshParameters['distmax'],  # Max distance  
            )

            geom.set_background_mesh([field0], operator="Min")
            mesh = geom.generate_mesh()
            points = mesh.points[:, :2]  # Get only x and y coordinates
        
        points = points[(points[:, 0] >= self.xMin) & (points[:, 0] <= self.xMax)]
        sdf = np.array([self.signed_distance(p, geo) for p in points])

        return points, sdf
    
    def getMaskPolygon(self, Lo, Ho):
        # Create points for masking
        Wo = 0.1
        thichness = 0.05 
        margin = 0.01
        yu = 0.5 + margin
        y45 = yu - thichness
        usx = -Lo - thichness
        dsx = -Lo + Wo + thichness
        yl = yu - Ho - thichness

        bPoints = [
            [usx, y45], [-Lo, yu], 
            [-Lo + Wo, yu], [dsx, y45],
            [dsx, yl], [usx, yl]
        ]

        # Define polygon for masking
        mask_polygon = Polygon(bPoints)
        return mask_polygon, usx, dsx, yl
    
         
    def setBaseMesh(self):
        startTime = time.time()
        baseMesh = torch.load('./data/ffs/baseMesh/mesh_points.th', weights_only=True)
        # Load mesh with large obstacle
        name = 'DP600_906,25000000000648_0,55666666666666553_0,49966666666666859'
        obstacleMesh = torch.load(f'./data/ffs/preprocessed600/{name}/mesh_points.th', weights_only=True)
        parameterDef = {'name': str, 're': float, 'Lo': float, 'Ho': float}
        parametersObs = readParametersFromFileName(name, parameterDef)
        self.LoObs = parametersObs['Lo']
        self.HoObs = parametersObs['Ho']
        endTime = time.time()
        print(f"Loading mesh took {endTime - startTime:.4f} seconds.")

        startTime = time.time()
        mask_polygon, usx, dsx, y1 = self.getMaskPolygon(self.LoObs, self.HoObs)


        baseMesh_np = baseMesh.numpy()
        obstacleMesh_np = obstacleMesh.numpy()
        endTime = time.time()
        print(f"Numpy conv. took {endTime - startTime:.4f} seconds.")

        startTime = time.time()
        # Identify points in obstacleMesh that are inside the mask_polygon
        obstacle_points = [Point(p) for p in obstacleMesh_np]
        obstacle_inside_mask = np.array([mask_polygon.contains(pt) for pt in obstacle_points])

        

        self.obstacleMesh = obstacleMesh_np[obstacle_inside_mask]
        self.baseMesh = baseMesh_np[(baseMesh_np[:, 0] >= self.xMin) & (baseMesh_np[:, 0] <= self.xMax)]

        print('base mesh set')

    def modify_base_mesh(self, Lo, Ho):
        """
        Generate a mesh-based point cloud by updating a mesh from training data and compute the signed distance function (SDF).
        """
        geo = self.ffsGeo(Lo=Lo, Ho=Ho)

        if self.baseMesh is None:
            self.setBaseMesh()

        mask_polygon, usx, dsx, yl  = self.getMaskPolygon(Lo, Ho)

        # Move obstacleMesh to correct location
        obstacleMesh = self.obstacleMesh.copy()
        obstacleMesh[:, 0] += self.LoObs - Lo
        obstacleMesh[:, 1] += self.HoObs - Ho
        obstacleMesh = obstacleMesh[(obstacleMesh[:, 0] >= self.xMin) & (obstacleMesh[:, 0] <= self.xMax)]
        obstacleMesh = obstacleMesh[obstacleMesh[:, 1] <= 0.5]

        

        baseMesh = self.baseMesh.copy()
        maskClose = ((baseMesh[:, 0] >= usx) & (baseMesh[:, 0] <= dsx) & (baseMesh[:, 1] >= yl))
        baseMeshClose = baseMesh[maskClose]
        baseMesh = baseMesh[~maskClose]

        # startTime = time.time()
        # Identify points in baseMesh that are outside the mask_polygon
        base_points = [Point(p) for p in baseMeshClose]
        base_outside_mask = np.array([not mask_polygon.contains(pt) for pt in base_points])
        # Keep only points outside the mask in baseMesh
        baseMesh_filtered = baseMeshClose[base_outside_mask]
        # endTime = time.time()
        # print(f"Creating base mask and contins took {endTime - startTime:.4f} seconds.")

        updatedMesh_np = np.vstack((baseMesh, baseMesh_filtered, obstacleMesh))
        # updatedMesh_np = np.vstack((baseMesh_filtered, obstacleMesh))
        # updatedMesh_np = baseMesh_filtered

        updatedMesh_np = self.train_dataset.subsample(nrPoints=self.totalPoints, mesh_pos=updatedMesh_np, seed=0)
        
        # startTime = time.time()
        # Compute SDF for the updated mesh
        sdf = np.array([self.signed_distance(p, geo) for p in updatedMesh_np])
        # endTime = time.time()
        # print(f"Sdf calc took {endTime - startTime:.4f} seconds.")



        return updatedMesh_np, sdf

    def preprocess(self, re_value, Lo, Ho):
        torch.manual_seed(0)  # Set a fixed seed
        np.random.seed(0)     # Set a fixed seed for numpy
        """
        Preprocess the input data by generating the point cloud and normalizing features.
        """
        if self.useMesh=='gmsh' and self.meshParameters is not None:
            points, sdf = self.generate_mesh_points(
                Lo=Lo,
                Ho=Ho,
            )
        elif self.useMesh=='modify':
            points, sdf = self.modify_base_mesh(
                Lo=Lo,
                Ho=Ho
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

    def infer(self, parameter_sets, output_pos=None):
        """
        Perform inference for a set of parameters using the model.
        Collect predictions together for easier analysis.
        """
        if self.model is None or self.device is None:
            raise ValueError("Model and device must be assigned before calling infer.")
        if self.useMesh == 'gmsh' and self.meshParameters is None:
            raise ValueError("Mesh parameters must be provided when useMesh is gmsh.")

        all_predictions = []
        all_parameters = []
        all_points = []

        idx = 0
        total_start_time = time.time()  # Start timing the entire inference process

        for re_value, Lo, Ho in parameter_sets:
            step_start_time = time.time()  # Start timing this step

            # Preprocessing step
            preprocess_start_time = time.time()
            batch = self.preprocess(re_value, Lo, Ho, idx)
            preprocess_end_time = time.time()
            # print(f"Preprocessing for parameters (re={re_value}, Lo={Lo}, Ho={Ho}) took {preprocess_end_time - preprocess_start_time:.4f} seconds.")

            # Output position setup
            current_output_pos = output_pos if output_pos is not None else batch['output_pos']

            # Inference step
            inference_start_time = time.time()
            with torch.no_grad():
                y_hat = self.model(
                    input_feat=batch['input_feat'].to(self.device),
                    input_pos=batch['input_pos'].to(self.device),
                    supernode_idxs=batch['supernode_idxs'].to(self.device),
                    batch_idx=batch['batch_idx'].to(self.device),
                    output_pos=current_output_pos.to(self.device),
                    re=batch['re'].to(self.device),
                )
            inference_end_time = time.time()
            # print(f"Inference for parameters (re={re_value}, Lo={Lo}, Ho={Ho}) took {inference_end_time - inference_start_time:.4f} seconds.")

            # Postprocessing step
            postprocess_start_time = time.time()
            all_points.append(self.train_dataset.denormalize_pos(current_output_pos.squeeze()))
            all_predictions.append(self.train_dataset.denormalize_feat(y_hat.cpu()))
            all_parameters.append({'re': re_value, 'Lo': Lo, 'Ho': Ho})
            postprocess_end_time = time.time()
            # print(f"Postprocessing for parameters (re={re_value}, Lo={Lo}, Ho={Ho}) took {postprocess_end_time - postprocess_start_time:.4f} seconds.")

            # Step timing
            step_end_time = time.time()
            # print(f"Total time for step {idx + 1} (re={re_value}, Lo={Lo}, Ho={Ho}) took {step_end_time - step_start_time:.4f} seconds.")

            idx += 1

        total_end_time = time.time()  # End timing the entire inference process
        # print(f"Total inference process took {total_end_time - total_start_time:.4f} seconds.")

        return {
            'parameters': all_parameters,
            'points': torch.stack(all_points),
            'predictions': torch.stack(all_predictions)
        }


    def get_batches(self, parameter_sets):
        """
        Generate and return a list of batches (size 1) for the given parameter sets.
        This method does not require a model or device to be assigned.
        """
        if self.useMesh=='gmsh' and self.meshParameters is None:
            raise ValueError("Mesh parameters must be provided when useMesh is gmsh.")
        
        batches = []
        idx = 0
        for re_value, Lo, Ho in parameter_sets:
            batch = self.preprocess(re_value, Lo, Ho, idx)
            batches.append(batch)
            idx += 1
        return batches