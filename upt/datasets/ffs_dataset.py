import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from data.ffs.readParameters import readParametersFromFileName
import pygmsh
from shapely.geometry import Point, LineString, Polygon
import numpy as np
import time 
from torch.utils.data import DataLoader

class ffsDataset(Dataset):
    def __init__(
            self,
            root,
            num_inputs,
            num_outputs,
            mode,
            crop_values=None,
            parameter_sets=None,
            use_inferencer_inputs=False,
            useMesh=None,
            meshParameters=None,
            customOutputPos=None,
            deterministic=False
    ):
        super().__init__()
        self.root = Path(root).expanduser()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mode = mode
        self.crop_values = crop_values
        self.parameter_sets = parameter_sets
        self.use_inferencer_inputs = use_inferencer_inputs
        self.useMesh = useMesh
        self.meshParameters = meshParameters
        self.obstacleMesh = None
        self.baseMesh = None
        self.baseSdf = None
        self.customOutputPos = customOutputPos
        self.deterministic = deterministic

        # Define spatial min/max of simulation
        if self.crop_values is None:
            normCoord = torch.load(self.root / 'coords_norm.th', weights_only=True)
            self.domain_min = normCoord['min_coords']
            self.domain_max = normCoord['max_coords']
        else:
            self.domain_min = torch.tensor(self.crop_values[0]).squeeze(0)
            self.domain_max = torch.tensor(self.crop_values[1]).squeeze(0)
        self.scale = 200

        self.xMin = self.domain_min[0].numpy()
        self.xMax = self.domain_max[0].numpy()
        


        # Mean/std for normalization
        normVars = torch.load(self.root / 'vars_norm.th', weights_only=True)
        self.mean = normVars['mean']
        self.std = normVars['std']

        self.uris = []
        # self.TEST_INDICES = []
        # self.parameterDef = {'name': str, 're': float, 'Lo': float, 'Ho': float}
        self.parameterDef = {'name': str, 're': float, 'Lo': float, 'Ho': float, 'dCp': float}
            
        self.TEST_INDICES = [445, 383, 382, 521, 163, 403, 143, 344, 487, 375, 338, 432, 472,  53,
        451, 510,  78, 280,  62, 552,  50, 207, 282, 532, 291,  76, 332, 257,
        489, 471,  38, 481, 151, 543, 401, 318, 167, 346,  92, 269,  85, 349,
        587,  77,  55, 465, 562, 442, 161, 503, 364, 144, 412, 210, 231, 513,
        295, 566, 217, 505, 406,  46, 501, 104, 365, 233,   0, 522,  11, 112,
        28, 461, 308,  40, 572, 573, 565, 476, 293, 334, 111, 102, 398, 497,
        91, 485, 317, 379, 370, 421, 260, 596, 141, 351, 423, 137,   9, 395,
        578, 292, 553, 429, 147,  49, 544, 561, 214, 400, 183, 186, 343, 244,
        281, 170, 277,  12, 134, 139, 106, 366]
        
        for name in sorted(os.listdir(self.root)):
            sampleDir = self.root / name
            if sampleDir.is_dir():
                self.uris.append(sampleDir)
                # dp = sampleDir.name.split('_')[0].replace('DP', '')
                # if int(dp) > 100:
                #     self.TEST_INDICES.append(len(self.uris)-1)
        
        # split into train/test uris
        if self.mode == "train":
            train_idxs = [i for i in range(len(self.uris)) if i not in self.TEST_INDICES]
            self.uris = [self.uris[train_idx] for train_idx in train_idxs]
            self.num_values = len(self.uris)
        elif self.mode == "test":
            self.uris = [self.uris[test_idx] for test_idx in self.TEST_INDICES]
            self.num_values = len(self.uris)
        elif self.mode == 'inference' and self.parameter_sets is not None:
            self.uris = None  # No URIs needed for parameter sets
            self.num_values = len(self.parameter_sets)
        else:
            raise NotImplementedError
        

    def __len__(self):
        return self.num_values

    def ffsGeo(self, Lo, Ho):
            # xMax = 12 # non SST
            # xMin = -6 # non SST
            xMax = 12 # SST
            xMin = -12 # SST
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
        if self.useMesh == None:
            return d if shape['geo'].contains(pt) else -d
        else:
            return d
            
    def generate_grid_points(self, Lo, Ho):
        """
        Generate a grid-based point cloud and compute the signed distance function (SDF).
        """
        if self.num_inputs == float("inf"):
            num_points = 2.5*10**4  # approx no points in mesh
        else:
            num_points = self.num_inputs  
        num_points *=(3/2) # increase no points to compensate for the ones removed by the step. UPDATE for SST if used
        
        aspect = (self.xMax - self.xMin) / 1
        Ny = int(np.sqrt(num_points / aspect))
        Nx = int(num_points / Ny)

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
        baseMesh = torch.load('./data/ffs/baseMesh/mesh_points.th', weights_only=True)
        baseSdf = torch.load('./data/ffs/baseMesh/mesh_sdf.th', weights_only=True)
        # Load mesh with large obstacle
        name = 'DP600_906,2500000000064_0,5566666666666655_0,4996666666666685_9,63446'
        obstacleMesh = torch.load(f'./data/ffs/preprocessed600/{name}/mesh_points.th', weights_only=True)
        parameterDef = {'name': str, 're': float, 'Lo': float, 'Ho': float}
        parametersObs = readParametersFromFileName(name, parameterDef)
        self.LoObs = parametersObs['Lo']
        self.HoObs = parametersObs['Ho']

        mask_polygon, usx, dsx, yl = self.getMaskPolygon(self.LoObs, self.HoObs)

        baseMesh_np = baseMesh.numpy()
        obstacleMesh_np = obstacleMesh.numpy()
        baseSdf_np = baseSdf.numpy()

        # Identify points in obstacleMesh that are inside the mask_polygon
        # obstacle_points = [Point(p) for p in obstacleMesh_np]
        # obstacleMask = np.array([mask_polygon.contains(pt) for pt in obstacle_points])

        obstacleMask = ((obstacleMesh_np[:, 0] >= usx) & (obstacleMesh_np[:, 0] <= dsx) & (obstacleMesh_np[:, 1] >= yl))
        baseMask = (baseMesh_np[:, 0] >= self.xMin) & (baseMesh_np[:, 0] <= self.xMax)

        self.obstacleMesh = obstacleMesh_np[obstacleMask]
        self.baseMesh = baseMesh_np[baseMask]
        self.baseSdf = baseSdf_np[baseMask]

        print('base mesh set')

    def modify_base_mesh(self, Lo, Ho):
        """
        Generate a mesh-based point cloud by updating a mesh from training data and compute the signed distance function (SDF).
        """
        geo = self.ffsGeo(Lo=Lo, Ho=Ho)

        # startTime = time.time()
        if self.baseMesh is None:
            self.setBaseMesh()
        # print(f"Set base mesh took {time.time() - startTime:.4f} seconds.")

        mask_polygon, usx, dsx, yl  = self.getMaskPolygon(Lo, Ho)

        # startTime = time.time()
        # Move obstacleMesh to correct location
        obstacleMesh = self.obstacleMesh.copy()
        obstacleMesh[:, 0] += self.LoObs - Lo
        obstacleMesh[:, 1] += self.HoObs - Ho
        obstacleMesh = obstacleMesh[(obstacleMesh[:, 0] >= self.xMin) & (obstacleMesh[:, 0] <= self.xMax)]
        obstacleMesh = obstacleMesh[obstacleMesh[:, 1] <= 0.5]
        obstacleSdf = np.ones(len(obstacleMesh))
        # obstacleSdf = np.array([self.signed_distance(p, geo) for p in obstacleMesh])
        # print(f"Moving obstacle mesh (and calculating sdf) took {time.time() - startTime:.4f} seconds.")
        
        # startTime = time.time()
        baseMesh = self.baseMesh.copy()
        baseSdf = self.baseSdf.copy()
        # maskClose = ((baseMesh[:, 0] >= usx-0.05) & (baseMesh[:, 0] <= dsx+0.05) & (baseMesh[:, 1] >= yl-0.05))
        maskClose = ((baseMesh[:, 0] >= usx) & (baseMesh[:, 0] <= dsx) & (baseMesh[:, 1] >= yl))
        # baseMeshClose = baseMesh[maskClose]
        # baseSdfClose = baseSdf[maskClose]
        baseMesh = baseMesh[~maskClose]
        baseSdf = baseSdf[~maskClose]
        # print(f"Masking base mesh arounf obstacle took {time.time() - startTime:.4f} seconds.")

        # startTime = time.time()
        # Identify points in baseMesh that are outside the mask_polygon
        # base_points = [Point(p) for p in baseMeshClose]
        # base_outside_mask = np.array([not mask_polygon.contains(pt) for pt in base_points])
        # print(f"Creating mask for close base mesh using polygon took {time.time() - startTime:.4f} seconds.")
        
        # startTime = time.time()
        # Keep only points outside the mask in baseMesh
        # baseMesh_filtered = baseMeshClose[base_outside_mask]
        # baseSdf_filtered = baseSdfClose[base_outside_mask]
        # baseSdf_filtered = np.array([self.signed_distance(p, geo) for p in baseMesh_filtered])
        # print(f"Masking close base mesh using polygon took {time.time() - startTime:.4f} seconds.")


        # startTime = time.time()
        # updatedMesh_np = np.vstack((baseMesh, baseMesh_filtered, obstacleMesh))
        # updatedSdf_np = np.hstack((baseSdf, baseSdf_filtered, obstacleSdf))
        updatedMesh_np = np.vstack((baseMesh, obstacleMesh))
        updatedSdf_np = np.hstack((baseSdf, obstacleSdf))        # print(f"Stacking took {time.time() - startTime:.4f} seconds.")

        if self.num_outputs != float("inf"):
            # updatedMesh_np = self.subsample(nrPoints=self.num_inputs, mesh_pos=updatedMesh_np, seed=0) #//NOTE// seed?
            updatedMesh_np, updatedSdf_np = self.subsample(nrPoints=self.num_inputs, mesh_pos=updatedMesh_np, features=updatedSdf_np, seed=0) #//NOTE// seed?
        
        # Update SDF around obstacle
        # startTime = time.time()
        maskClose = ((updatedMesh_np[:, 0] >= usx-0.5) & (updatedMesh_np[:, 0] <= dsx+0.5))
        points_close = updatedMesh_np[maskClose]
        updatedSdf_np[maskClose] = np.array([self.signed_distance(p, geo) for p in points_close])
        # print(f"Updating SDF around obstacle took {time.time() - startTime:.4f} seconds.")

        return updatedMesh_np, updatedSdf_np

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

        input_feat = self.normalize_sdf(torch.tensor(sdf, dtype=torch.float32)).unsqueeze(1)
        input_pos = self.normalize_pos(torch.tensor(points, dtype=torch.float32))
        re = self.normalize_re(torch.tensor([re_value], dtype=torch.float32)).squeeze(0)

        return {
            'input_feat': input_feat,
            'input_pos': input_pos,
            'output_pos': input_pos,
            're': re
        }

    def normalize_pos(self, pos):
        pos = pos.sub(self.domain_min).div(self.domain_max - self.domain_min).mul(self.scale)
        assert torch.all(0 <= pos)
        assert torch.all(pos <= self.scale)
        return pos

    def denormalize_pos(self, pos):
        pos = pos.div(self.scale).mul(self.domain_max - self.domain_min).add(self.domain_min)
        return pos

    def normalize_feat(self, feat):
        feat = feat.sub(self.mean[:3]).div(self.std[:3])
        return feat

    def denormalize_feat(self, feat):
        feat = feat.mul(self.std[:3]).add(self.mean[:3])
        return feat

    def normalize_sdf(self, sdf):
        sdf = sdf.sub(self.mean[-2]).div(self.std[-2])
        return sdf

    def denormalize_sdf(self, sdf):
        sdf = sdf.mul(self.std[-2]).add(self.mean[-2])
        return sdf

    def normalize_re(self, re):
        # re = (re - 550) / 260
        
        # min max scaling
        re = ((re - 25000) / (35000 - 25000)) * self.scale
        # re = ((re - 25000) / (35000 - 25000)) * 1
        
        # mean/std scaling
        # re = ((re -self.mean[-1]) / self.std[-1])
        
        # mean/std scaling with self.scale
        # re = ((re -self.mean[-1]) / self.std[-1]) * self.scale
        
        return re

    def denormalize_re(self, re):
        # re = (re * 260) + 550
        
        # min max scaling
        re = (re / self.scale) * (35000 - 25000) + 25000
        # re = (re / 1) * (35000 - 25000) + 25000
        
        # mean/std scaling
        # re =  (re * self.std[-1]) + self.mean[-1]
        
        # mean/std scaling with self.scale
        # re =  ((re/self.scale) * self.std[-1]) + self.mean[-1]
        return re

    def subsample(self, nrPoints, mesh_pos, features=None, seed=None):
        if seed is None:
            rng = None
        else:
            rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(mesh_pos), generator=rng)[:nrPoints]
        if features is not None:
            return mesh_pos[perm], features[perm]
        else:
            return mesh_pos[perm]
        
    def setInferenceMode(self, parameter_sets):
        self.mode = 'inference'
        self.parameter_sets = parameter_sets
        self.num_values = len(self.parameter_sets)         

    def __getitem__(self, idx):
        if self.mode == "inference":
            # Handle inference mode with parameter sets
            re_value, Lo, Ho = self.parameter_sets[idx]
            input_data = self.preprocess(re_value, Lo, Ho)
            if self.customOutputPos is not None:
                # Use custom output positions
                output_pos = self.customOutputPos
                output_pos = self.normalize_pos(output_pos)
            else:
                # Use the same output positions as input positions
                output_pos = input_data['output_pos']
            return dict(
                index=idx,
                input_feat=input_data['input_feat'],
                input_pos=input_data['input_pos'],
                target_feat=None,
                output_pos=output_pos,
                re=input_data['re'],
                name=f"param_set_{idx}",
                dCp=None
            )

        # Training/test mode
        mesh_pos = torch.load(self.uris[idx] / "mesh_points.th", weights_only=True)
        u = torch.load(self.uris[idx] / "u.th", weights_only=True)
        v = torch.load(self.uris[idx] / "v.th", weights_only=True)
        p = torch.load(self.uris[idx] / "p.th", weights_only=True)
        target = torch.cat((u, v, p), dim=1)
        parameters = readParametersFromFileName(self.uris[idx].name, self.parameterDef)
        re = torch.tensor(parameters['re'], dtype=torch.float32).squeeze(0)
        Lo = parameters['Lo']
        Ho = parameters['Ho']
        dCp = torch.tensor(parameters['dCp'], dtype=torch.float32).squeeze(0)
        sdf = torch.load(self.uris[idx] / "mesh_sdf.th", weights_only=True).unsqueeze(1).float()
       
        if self.crop_values is not None:
            # Filter mesh_pos, input_feat and target based on self.domain_min and self.domain_max
            mask = (mesh_pos[:, 0] >= self.domain_min[0]) & (mesh_pos[:, 0] <= self.domain_max[0]) & \
                (mesh_pos[:, 1] >= self.domain_min[1]) & (mesh_pos[:, 1] <= self.domain_max[1])
            mesh_pos = mesh_pos[mask]
            target = target[mask]
            sdf = sdf[mask]

        # Normalize
        mesh_pos = self.normalize_pos(mesh_pos)
        target = self.normalize_feat(target)
        re = self.normalize_re(re)
        sdf = self.normalize_sdf(sdf)

        # Subsample outputs
        # output_pos, target_feat = self.subsample(self.num_outputs, mesh_pos, target, seed=idx + 1)
        if self.num_outputs != float("inf"):
            if self.mode == "train" and not self.deterministic:
                seed = None
            else:
                seed = idx +1
            output_pos, target_feat = self.subsample(nrPoints=self.num_outputs, mesh_pos=mesh_pos, features=target, seed=seed)
        else:
            target_feat = target
            output_pos = mesh_pos #.clone()



        # Subsample inputs
        if self.use_inferencer_inputs:
            # Use input_pos and input_feat from the inferencer
            input_data = self.preprocess(re, Lo, Ho)
            input_pos = input_data['input_pos']
            input_feat = input_data['input_feat']
        else:
            # Subsample inputs from the current data point
            # input_pos, input_feat = self.subsample(self.num_inputs, mesh_pos, sdf, seed=idx)
            if self.num_inputs != float("inf") and not self.deterministic:            
                if self.mode == "train":
                        seed = None
                else:
                    seed = idx
                input_pos, input_feat = self.subsample(nrPoints=self.num_inputs, mesh_pos=mesh_pos, features=sdf, seed=seed)
            else:
                input_feat = sdf
                input_pos = mesh_pos #.clone()

        if self.customOutputPos is not None:
            # Use custom output positions
            output_pos = self.customOutputPos
            output_pos = self.normalize_pos(output_pos)
            target_feat = None

        
        return dict(
            index=idx,
            input_feat=input_feat,
            input_pos=input_pos,
            target_feat=target_feat,
            output_pos=output_pos,
            re=re,
            name=str(self.uris[idx].name),
            dCp=dCp
        )