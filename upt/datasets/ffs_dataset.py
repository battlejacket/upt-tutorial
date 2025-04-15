import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ffsDataset(Dataset):
    def __init__(
            self,
            root,
            # how many input points to sample
            num_inputs,
            # how many output points to sample
            num_outputs,
            # train or test
            mode,
            crop_values = None,
            # crop_values = [[x_min, y_min], [x_max, y_max]],
    ):
        super().__init__()
        root = Path(root).expanduser()
        self.root = root
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mode = mode
        self.crop_values = crop_values

        # define spatial min/max of simulation (for normalizing to [0, 1] and then scaling to [0, 200] for pos_embed)
        if self.crop_values is None:
            normCoord = torch.load(self.root / 'coords_norm.th', weights_only=True)
            self.domain_min = normCoord['min_coords']
            self.domain_max = normCoord['max_coords']
        else:
            self.domain_min = torch.tensor(self.crop_values[0]).squeeze(0)
            self.domain_max = torch.tensor(self.crop_values[1]).squeeze(0)
        self.scale = 200

        # mean/std for normalization (calculated during preprocessing) (u,v,p,sdf)
        normVars = torch.load(self.root / 'vars_norm.th', weights_only=True)
        self.mean = normVars['mean']
        self.std = normVars['std']

        self.uris = []
        self.TEST_INDICES = []
        for name in sorted(os.listdir(self.root)):
            sampleDir = self.root / name
            if sampleDir.is_dir():
                self.uris.append(sampleDir)
                if int(name.split("_")[0].replace('DP', '')) > 100:
                    self.TEST_INDICES.append(len(self.uris)-1)
        
        # split into train/test uris
        if self.mode == "train":
            train_idxs = [i for i in range(len(self.uris)) if i not in self.TEST_INDICES]
            self.uris = [self.uris[train_idx] for train_idx in train_idxs]
        elif self.mode == "test":
            self.uris = [self.uris[test_idx] for test_idx in self.TEST_INDICES]
        else:
            raise NotImplementedError
        
        self.num_values = len(self.uris)

    def __len__(self):
        return len(self.uris)

    def normalize_pos(self, pos):
        # normalize the position
        pos = pos.sub_(self.domain_min).div_(self.domain_max - self.domain_min).mul_(self.scale)
        assert torch.all(0 <= pos)
        assert torch.all(pos <= self.scale)
        return pos
    
    def denormalize_pos(self, pos):
        # denormalize the position
        pos = pos.div_(self.scale).mul_(self.domain_max - self.domain_min).add_(self.domain_min)
        return pos

    def normalize_feat(self, feat):
        # normalize the prediction
        feat = feat.sub_((self.mean[:-1])).div_((self.std[:-1]))
        return feat
    
    def denormalize_feat(self, feat):
        # denormalize the prediction
        feat = feat.mul_((self.std[:-1])).add_((self.mean[:-1]))
        return feat
    
    def normalize_sdf(self, sdf):
        sdf = sdf.sub_((self.mean[:-1])).div_((self.std[:-1]))
        # sdf = (sdf - self.mean[-1]) / self.std[-1]
        return sdf
        
    def denormalize_sdf(self, sdf):
        sdf = sdf.mul_((self.std[:-1])).add_((self.mean[:-1]))
        # sdf = sdf*self.std[-1] + self.mean[-1]
        return sdf

    def __getitem__(self, idx):
        # load mesh points and targets (u, v, p)
        mesh_pos = torch.load(self.uris[idx] / "mesh_points.th", weights_only=True)
        u = torch.load(self.uris[idx] / "u.th", weights_only=True)
        v = torch.load(self.uris[idx] / "v.th", weights_only=True)
        p = torch.load(self.uris[idx] / "p.th", weights_only=True)
        target = torch.cat((u, v, p), dim=1)
        re = float(str(self.uris[idx]).split('/')[-1].split('-')[0].split('_')[-1].replace(',', '.'))
        sdf = torch.load(self.uris[idx] / "mesh_sdf.th", weights_only=True).unsqueeze(1).float()
        # sdf = torch.ones_like(sdf)
        # sdf = torch.zeros_like(sdf)
        
        if self.crop_values is not None:
            # Filter mesh_pos, input_feat and target based on self.domain_min and self.domain_max
            mask = (mesh_pos[:, 0] >= self.domain_min[0]) & (mesh_pos[:, 0] <= self.domain_max[0]) & \
                (mesh_pos[:, 1] >= self.domain_min[1]) & (mesh_pos[:, 1] <= self.domain_max[1])
            mesh_pos = mesh_pos[mask]
            target = target[mask]
            sdf = sdf[mask]

        # normalize
        mesh_pos = self.normalize_pos(mesh_pos)
        target = self.normalize_feat(target)
        re = (re - 550) / 260
        sdf = self.normalize_sdf(sdf)

        # subsample random input pixels (locations of inputs and outputs does not have to be the same)
        if self.num_inputs != float("inf"):
            if self.mode == "train":
                rng = None
            else:
                rng = torch.Generator().manual_seed(idx)
            
            input_perm = torch.randperm(len(mesh_pos), generator=rng)[:self.num_inputs]
            input_feat = sdf[input_perm]
            input_pos = mesh_pos[input_perm] #.clone()
        else:
            input_feat = sdf
            input_pos = mesh_pos #.clone()

        # subsample random output pixels (locations of inputs and outputs does not have to be the same)
        if self.num_outputs != float("inf"):
            if self.mode == "train":
                rng = None
            else:
                rng = torch.Generator().manual_seed(idx + 1)
            output_perm = torch.randperm(len(target), generator=rng)[:self.num_outputs]
            target_feat = target[output_perm]
            output_pos = mesh_pos[output_perm] #.clone()
        else:
            target_feat = target
            output_pos = mesh_pos #.clone()
        
        return dict(
            index=idx,
            input_feat=input_feat,
            input_pos=input_pos,
            target_feat=target_feat,
            output_pos=output_pos,
            re=re,
            name=str(self.uris[idx]).split('/')[-1],
        )