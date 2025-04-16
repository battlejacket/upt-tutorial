from shapely.geometry import Point, LineString, Polygon
import numpy as np
import torch


class ffsInference:
    def __init__(self, model, train_dataset, totalPoints, numSupernodes, device, xMin = None, xMax = None):
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
        geo = Polygon(bPoints)
        boundary = LineString(bPoints)
        return {'boundary': boundary, 'geo': geo}

    def signed_distance(self, p, shape):
        pt = Point(p)
        d = pt.distance(shape['boundary'])
        return d if shape['geo'].contains(pt) else -d

    def evalPointCloud(self, xMin, xMax, Lo, Ho):
        # Generate the full point cloud
        aspect = (xMax - xMin) / 1
        Ny = int(np.sqrt(self.totalPoints / aspect))
        Nx = int(self.totalPoints / Ny)

        x = np.linspace(xMin, xMax, Nx)
        y = np.linspace(-0.5, 0.5, Ny)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)

        # Compute SDF
        geo = self.ffsGeo(Lo=Lo, Ho=Ho)
        sdf = np.array([self.signed_distance(p, geo) for p in points])

        mask = (sdf >= 0)
        insidePoints = points[mask]
        insideSdf = sdf[mask]

        return {
            'input_feat': torch.tensor(insideSdf, dtype=torch.float32),
            'input_pos': torch.tensor(insidePoints, dtype=torch.float32)
        }

    def preprocess(self, re_value, Lo, Ho, idx):
        pointCloud = self.evalPointCloud(
            xMin=self.xMin,
            xMax=self.xMax,
            Lo=Lo,
            Ho=Ho
        )
        input_feat = self.train_dataset.normalize_sdf(pointCloud['input_feat'])
        # input_len = len(input_feat) 
        input_pos = self.train_dataset.normalize_pos(pointCloud['input_pos'])
        re = self.train_dataset.normalize_re(torch.tensor([re_value]))
        supernode_idxs = torch.randperm(len(input_feat))[:self.numSupernodes]
        # batch_idx = torch.full((input_len,), idx, dtype=torch.int32)
        batch_idx = torch.zeros(len(input_feat), dtype=torch.int32)
        return {
            'input_feat': input_feat.unsqueeze(1),
            'input_pos': input_pos,
            'supernode_idxs': supernode_idxs,
            'batch_idx': batch_idx,
            'output_pos': input_pos.unsqueeze(0),
            're': re
        }

    
    def infer(self, parameter_sets, output_pos=None):
        results = []
        idx = 0
        for re_value, Lo, Ho in parameter_sets:
            batch = self.preprocess(re_value, Lo, Ho, idx)
            
            # Ensure output_pos is unique for each parameter set
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
                result = dict(parameters={ 're': re_value, 'Lo': Lo, 'Ho': Ho},
                            points=self.train_dataset.denormalize_pos(current_output_pos.squeeze()),
                            prediction=self.train_dataset.denormalize_feat(y_hat.cpu()))
                results.append(result)
                idx += 1
        return results