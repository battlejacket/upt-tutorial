import einops
import scipy
import os
import shutil

import meshio
import numpy as np
import torch
from kappautils.param_checking import to_3tuple, to_2tuple
from torch_geometric.nn.pool import radius, radius_graph

from distributed.config import barrier, is_data_rank0
from .base.dataset_base import DatasetBase


class ffsDataset(DatasetBase):
    # generated with torch.randperm(889, generator=torch.Generator().manual_seed(0))[:189]
    TEST_INDICES = []

    def __init__(
            self,
            split,
            radius_graph_r=None,
            radius_graph_max_num_neighbors=None,
            num_input_points_ratio=None,
            num_query_points_ratio=None,
            grid_resolution=None,
            num_supernodes=None,
            standardize_query_pos=False,
            concat_pos_to_sdf=False,
            global_root=None,
            local_root=None,
            seed=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.split = split
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors or int(1e10)
        self.num_supernodes = num_supernodes
        self.seed = seed
        if num_input_points_ratio is None:
            self.num_input_points_ratio = None
        else:
            self.num_input_points_ratio = to_2tuple(num_input_points_ratio)
        self.num_query_points_ratio = num_query_points_ratio
        if grid_resolution is not None:
            self.grid_resolution = to_3tuple(grid_resolution)
        else:
            self.grid_resolution = None


        global_root, local_root = self._get_roots(global_root, local_root, "ffs_dataset")


        # define spatial min/max of simulation (for normalizing to [0, 1] and then scaling to [0, 200] for pos_embed)
        normCoord = torch.load(global_root / 'preprocessed/coords_norm.th')
        self.domain_min = normCoord['min_coords']
        self.domain_max = normCoord['max_coords']
        self.scale = 200
        self.standardize_query_pos = standardize_query_pos
        self.concat_pos_to_sdf = concat_pos_to_sdf

        # mean/std for normalization (calculated on the train samples)
        normVars = torch.load(global_root / 'preprocessed/vars_norm.th')
        self.mean = normVars['mean']
        self.std = normVars['std']

        # source_root
        if local_root is None:
            # load data from global_root
            self.source_root = global_root / "preprocessed"
            self.logger.info(f"data_source (global): '{self.source_root}'")
        else:
            # load data from local_root
            self.source_root = local_root / "FFS"
            if is_data_rank0():
                # copy data from global to local
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{self.source_root}'")
                if not self.source_root.exists():
                    self.logger.info(
                        f"copying {(global_root / 'preprocessed').as_posix()} "
                        f"to {(self.source_root / 'preprocessed').as_posix()}"
                    )
                    shutil.copytree(global_root / "preprocessed", self.source_root / "preprocessed")
            self.source_root = self.source_root / "preprocessed"
            barrier()
        assert self.source_root.exists(), f"'{self.source_root.as_posix()}' doesn't exist"
        assert self.source_root.name == "preprocessed", f"'{self.source_root.as_posix()}' is not preprocessed folder"

        # discover uris
        self.uris = []
        for name in sorted(os.listdir(self.source_root)):
            sampleDir = self.source_root / name
            if sampleDir.is_dir():
                self.uris.append(sampleDir)
                if int(name.split("_")[0].replace('DP', '')) > 100:
                    self.TEST_INDICES.append(len(self.uris))
        
        # split into train/test uris
        if split == "train":
            train_idxs = [i for i in range(len(self.uris)) if self.uris[i] not in self.TEST_INDICES]
            self.uris = [self.uris[train_idx] for train_idx in train_idxs]
            # assert len(self.uris) == 700
        elif split == "test":
            self.uris = [self.uris[test_idx] for test_idx in self.TEST_INDICES]
            # assert len(self.uris) == 20
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.uris)

    # get output variables
    def getitem_u(self, idx, ctx=None):
        u = torch.load(self.uris[idx] / "u.th")
        u -= self.mean[0]
        u /= self.std[0]
        return u
    
    def getitem_v(self, idx, ctx=None):
        v = torch.load(self.uris[idx] / "v.th")
        v -= self.mean[1]
        v /= self.std[1]
        return v
    
    def getitem_p(self, idx, ctx=None):
        p = torch.load(self.uris[idx] / "p.th")
        p -= self.mean[2]
        p /= self.std[2]
        return p
    
    def getitem_target(self, idx, ctx=None):
        u = self.getitem_u(idx, ctx)
        v = self.getitem_v(idx, ctx)
        p = self.getitem_p(idx, ctx)
        target = torch.cat((u, v, p), dim=1)
        return target
    
    def getitem_Re(self, idx, ctx=None):
        self.uris
        re = float(str(self.uris[idx]).split('/')[-1].split('-')[0].split('_')[-1])
        re -= 550
        re /= 260
        return re

    def getitem_mesh_pos(self, idx, ctx=None):
        if ctx is not None and "mesh_pos" in ctx:
            return ctx["mesh_pos"]
        mesh_pos = self.getitem_all_pos(idx, ctx=ctx)
        # sample mesh points
        if self.num_input_points_ratio is not None:
            if self.split == "test":
                assert self.seed is not None
            if self.seed is not None:
                # deterministically downsample for evaluation
                generator = torch.Generator().manual_seed(self.seed + int(idx))
            else:
                generator = None
            # get number of samples
            if self.num_input_points_ratio[0] == self.num_input_points_ratio[1]:
                # fixed num_input_points_ratio
                end = int(len(mesh_pos) * self.num_input_points_ratio[0])
            else:
                # variable num_input_points_ratio
                lb, ub = self.num_input_points_ratio
                num_input_points_ratio = torch.rand(size=(1,), generator=generator).item() * (ub - lb) + lb
                end = int(len(mesh_pos) * num_input_points_ratio)
            # uniform sampling
            perm = torch.randperm(len(mesh_pos), generator=generator)[:end]
            mesh_pos = mesh_pos[perm]
        if ctx is not None:
            ctx["mesh_pos"] = mesh_pos
        return mesh_pos

    def getitem_all_pos(self, idx, ctx=None):
        if ctx is not None and "all_pos" in ctx:
            return ctx["all_pos"]
        all_pos = torch.load(self.uris[idx] / "mesh_points.th")
        # rescale for sincos positional embedding
        all_pos.sub_(self.domain_min).div_(self.domain_max - self.domain_min).mul_(self.scale)
        assert torch.all(0 <= all_pos)
        assert torch.all(all_pos <= self.scale) #!!
        if ctx is not None:
            ctx["all_pos"] = all_pos
        return all_pos

    def getitem_query_pos(self, idx, ctx=None):
        if ctx is not None and "query_pos" in ctx:
            return ctx["query_pos"]
        query_pos = self.getitem_all_pos(idx, ctx=ctx)
        # sample query points
        if self.num_query_points_ratio is not None:
            if self.split == "test":
                assert self.seed is not None
            if self.seed is not None:
                # deterministically downsample for evaluation
                generator = torch.Generator().manual_seed(self.seed + int(idx))
            else:
                generator = None
            # get number of samples
            end = int(len(query_pos) * self.num_query_points_ratio)
            # uniform sampling
            perm = torch.randperm(len(query_pos), generator=generator)[:end]
            query_pos = query_pos[perm]
        # shift query_pos to [-1, 1] (required for torch.nn.functional.grid_sample)
        if self.standardize_query_pos:
            query_pos = query_pos / (self.scale / 2) - 1
        if ctx is not None:
            ctx["query_pos"] = query_pos
        return query_pos

    def _get_generator(self, idx):
        if self.split == "test":
            return torch.Generator().manual_seed(int(idx) + (self.seed or 0))
        if self.seed is not None:
            return torch.Generator().manual_seed(int(idx) + self.seed)
        return None

        # check for correctness of interpolation
        # import matplotlib.pyplot as plt
        # import os
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # plt.scatter(mesh_pos[:, 0], mesh_pos[:, 1])
        # plt.show()
        # plt.clf()
        # plt.imshow(grid.sum(dim=2).sum(dim=2), origin="lower")
        # plt.show()
        # plt.clf()
        # import torch.nn.functional as F
        # grid = einops.rearrange(grid, "h w d dim -> 1 dim h w d")
        # query_pos = self.getitem_query_pos(idx, ctx=ctx)
        # query_pos = einops.rearrange(query_pos, "num_points ndim -> 1 num_points 1 1 ndim")
        # mesh_values = F.grid_sample(input=grid, grid=query_pos, align_corners=False).squeeze(-1)
        # plt.scatter(*query_pos.squeeze().unbind(1), c=mesh_values[0, 0, :, 0])
        # plt.show()
        # plt.clf()

        return grid