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
    ):
        super().__init__()
        root = Path(root).expanduser()
        self.root = root
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mode = mode

        # define spatial min/max of simulation (for normalizing to [0, 1] and then scaling to [0, 200] for pos_embed)
        normCoord = torch.load(self.root / 'coords_norm.th', weights_only=True)
        self.domain_min = normCoord['min_coords']
        self.domain_max = normCoord['max_coords']
        self.scale = 200

        # mean/std for normalization (calculated on the train samples)
        normVars = torch.load(self.root / 'vars_norm.th', weights_only=True)
        self.mean = normVars['mean']
        self.std = normVars['std']

        # # discover simulations
        # self.case_names = list(sorted(os.listdir(root)))
        # self.num_timesteps = len(
        #     [
        #         fname for fname in os.listdir(root / self.case_names[0])
        #         if fname.endswith("_mesh.th")
        #     ],
        # )

        self.uris = []
        self.TEST_INDICES = []
        for name in sorted(os.listdir(self.root)):
            sampleDir = self.root / name
            if sampleDir.is_dir():
                self.uris.append(sampleDir)
                if int(name.split("_")[0].replace('DP', '')) > 100:
                    self.TEST_INDICES.append(len(self.uris))
        self.num_values = len(self.uris)

        # split into train/test uris
        if self.mode == "train":
            train_idxs = [i for i in range(len(self.uris)) if self.uris[i] not in self.TEST_INDICES]
            self.uris = [self.uris[train_idx] for train_idx in train_idxs]
        elif self.mode == "test":
            self.uris = [self.uris[test_idx] for test_idx in self.TEST_INDICES]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.uris)

    def getitem_u(self, idx):
        u = torch.load(self.uris[idx] / "u.th", weights_only=True)
        u -= self.mean[0]
        u /= self.std[0]
        return u
    
    def getitem_v(self, idx):
        v = torch.load(self.uris[idx] / "v.th", weights_only=True)
        v -= self.mean[1]
        v /= self.std[1]
        return v
    
    def getitem_p(self, idx):
        p = torch.load(self.uris[idx] / "p.th", weights_only=True)
        p -= self.mean[2]
        p /= self.std[2]
        return p
    
    def getitem_re(self, idx):
        re = float(str(self.uris[idx]).split('/')[-1].split('-')[0].split('_')[-1].replace(',', '.'))
        re -= 550
        re /= 260
        return re
    
    def getitem_all_pos(self, idx):
        all_pos = torch.load(self.uris[idx] / "mesh_points.th", weights_only=True)
        # rescale for sincos positional embedding
        all_pos.sub_(self.domain_min).div_(self.domain_max - self.domain_min).mul_(self.scale)
        assert torch.all(0 <= all_pos)
        assert torch.all(all_pos <= self.scale)
        return all_pos

    def __getitem__(self, idx):
        u = self.getitem_u(idx)
        v = self.getitem_v(idx)
        p = self.getitem_p(idx)
        re = self.getitem_re(idx)

        target = torch.cat((u, v, p), dim=1)

        mesh_pos = self.getitem_all_pos(idx)


        # subsample random input pixels (locations of inputs and outputs does not have to be the same)
        if self.num_inputs != float("inf"):
            if self.mode == "train":
                rng = None
            else:
                rng = torch.Generator().manual_seed(idx)
            input_perm = torch.randperm(len(mesh_pos), generator=rng)[:self.num_inputs]
            # input_feat = x[input_perm]
            input_pos = mesh_pos[input_perm].clone()
        else:
            # input_feat = x
            input_pos = mesh_pos.clone()

        # subsample random output pixels (locations of inputs and outputs does not have to be the same)
        if self.num_outputs != float("inf"):
            if self.mode == "train":
                rng = None
            else:
                rng = torch.Generator().manual_seed(idx + 1)
            output_perm = torch.randperm(len(target), generator=rng)[:self.num_outputs]
            target_feat = target[output_perm]
            output_pos = mesh_pos[output_perm].clone()
        else:
            target_feat = target
            output_pos = mesh_pos.clone()

        return dict(
            index=idx,
            # input_feat=input_feat,
            input_pos=input_pos,
            target_feat=target_feat,
            output_pos=output_pos,
            re=re,
        )