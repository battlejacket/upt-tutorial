import torch
from torch.utils.data import default_collate
from sklearn.neighbors import KernelDensity
import numpy as np

class ffsCollator:
    def __init__(self, num_supernodes, deterministic):
        self.num_supernodes = num_supernodes
        self.deterministic = deterministic

    def __call__(self, batch):
        collated_batch = {}

        # inputs to sparse tensors
        # position: batch_size * (num_inputs, ndim) -> (batch_size * num_inputs, ndim)
        # features: batch_size * (num_inputs, dim) -> (batch_size * num_inputs, dim)
        input_pos = []
        input_feat = []
        input_lens = []
        for i in range(len(batch)):
            pos = batch[i]["input_pos"]
            feat = batch[i]["input_feat"]
            assert len(pos) == len(feat)
            input_pos.append(pos)
            input_feat.append(feat)
            input_lens.append(len(pos))
        collated_batch["input_pos"] = torch.concat(input_pos)
        collated_batch["input_feat"] = torch.concat(input_feat)

        # select supernodes
        supernodes_offset = 0
        supernode_idxs = []
        for i in range(len(input_lens)):
            if self.deterministic:
                rng = torch.Generator().manual_seed(batch[i]["index"])
            else:
                rng = None
            perm = torch.randperm(len(input_pos[i]), generator=rng)[:self.num_supernodes] + supernodes_offset
            supernode_idxs.append(perm)
            supernodes_offset += input_lens[i]
        collated_batch["supernode_idxs"] = torch.concat(supernode_idxs)

        if False:
            # Combine all CFD training input points into a single array
            all_mesh_points = np.concatenate(all_training_mesh_points, axis=0)  # shape (N_total, 2)

            # Fit KDE
            kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(all_mesh_points)

            log_densities = kde.score_samples(inference_points)  # shape (N_points,)
            densities = np.exp(log_densities)
            
            probs = densities / np.sum(densities)

            num_supernodes = 128  # or whatever you normally use
            supernode_idxs = np.random.choice(
                len(inference_points),
                size=num_supernodes,
                replace=False,
                p=probs
            )

            # from scipy.stats import multivariate_normal

            # mu = [x_center, y_center]  # e.g., where the step is
            # cov = [[σx**2, 0], [0, σy**2]]

            # rv = multivariate_normal(mean=mu, cov=cov)
            # densities = rv.pdf(inference_points)
            # probs = densities / np.sum(densities)


        # create batch_idx tensor
        batch_idx = torch.empty(sum(input_lens), dtype=torch.long)
        start = 0
        cur_batch_idx = 0
        for i in range(len(input_lens)):
            end = start + input_lens[i]
            batch_idx[start:end] = cur_batch_idx
            start = end
            cur_batch_idx += 1
        collated_batch["batch_idx"] = batch_idx

        # output_pos
        collated_batch["output_pos"] = default_collate([batch[i]["output_pos"] for i in range(len(batch))])

        # target_feat to sparse tensor
        # batch_size * (num_outputs, dim) -> (batch_size * num_outputs, dim)
        if batch[i]['target_feat'] is not None:
            collated_batch["target_feat"] = torch.concat([batch[i]["target_feat"] for i in range(len(batch))])
        if batch[i]['dCp'] is not None:
            collated_batch["dCp"]= default_collate([batch[i]["dCp"] for i in range(len(batch))])
        
        
        collated_batch["re"] = default_collate([batch[i]["re"] for i in range(len(batch))])
        collated_batch["name"]= default_collate([batch[i]["name"] for i in range(len(batch))])
        return collated_batch
