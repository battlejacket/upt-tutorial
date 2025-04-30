import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from data.ffs.readParameters import readParametersFromFileName
import pygmsh


class ffsDataset(Dataset):
    def __init__(
            self,
            root,
            num_inputs,
            num_outputs,
            mode,
            crop_values=None,
            parameter_sets=None,
            use_inferencer_inputs=False,  # New flag to use inputs from the inferencer
    ):
        super().__init__()
        self.root = Path(root).expanduser()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mode = mode
        self.crop_values = crop_values
        self.parameter_sets = parameter_sets
        self.use_inferencer_inputs = use_inferencer_inputs

        # Define spatial min/max of simulation
        if self.crop_values is None:
            normCoord = torch.load(self.root / 'coords_norm.th', weights_only=True)
            self.domain_min = normCoord['min_coords']
            self.domain_max = normCoord['max_coords']
        else:
            self.domain_min = torch.tensor(self.crop_values[0]).squeeze(0)
            self.domain_max = torch.tensor(self.crop_values[1]).squeeze(0)
        self.scale = 200

        # Mean/std for normalization
        normVars = torch.load(self.root / 'vars_norm.th', weights_only=True)
        self.mean = normVars['mean']
        self.std = normVars['std']

        self.uris = []
        self.parameterDef = {'name': str, 're': float, 'Lo': float, 'Ho': float}

        # Load dataset URIs
        for name in sorted(os.listdir(self.root)):
            sampleDir = self.root / name
            if sampleDir.is_dir():
                self.uris.append(sampleDir)

        # Handle inference mode
        if self.mode == 'inference' and self.parameter_sets is not None:
            self.uris = None  # No URIs needed for parameter sets
            self.num_values = len(self.parameter_sets)
        else:
            self.num_values = len(self.uris)

    def __len__(self):
        return self.num_values

    def preprocess(self, re_value, Lo, Ho):
        """
        Preprocess data for inference.
        """
        # Generate input points and features
        input_pos, input_feat = self.generate_input_points(Lo, Ho)
        re = self.normalize_re(torch.tensor([re_value], dtype=torch.float32))

        # Generate output points (optional for inference)
        output_pos = input_pos.clone()  # Example: use the same points for output
        return {
            'input_feat': input_feat,
            'input_pos': input_pos,
            'output_pos': output_pos,
            're': re
        }

    def generate_input_points(self, Lo, Ho):
        """
        Generate input points and features based on Lo and Ho.
        """
        # Example implementation (can be grid-based or mesh-based)
        x = torch.linspace(self.domain_min[0], self.domain_max[0], self.num_inputs)
        y = torch.linspace(self.domain_min[1], self.domain_max[1], self.num_inputs)
        X, Y = torch.meshgrid(x, y)
        input_pos = torch.stack([X.flatten(), Y.flatten()], dim=1)
        input_feat = torch.zeros(input_pos.shape[0], 1)  # Example: placeholder features
        return input_pos, input_feat

    def normalize_pos(self, pos):
        pos = pos.sub(self.domain_min).div(self.domain_max - self.domain_min).mul(self.scale)
        assert torch.all(0 <= pos)
        assert torch.all(pos <= self.scale)
        return pos

    def denormalize_pos(self, pos):
        pos = pos.div(self.scale).mul(self.domain_max - self.domain_min).add(self.domain_min)
        return pos

    def normalize_feat(self, feat):
        feat = feat.sub(self.mean[:-1]).div(self.std[:-1])
        return feat

    def denormalize_feat(self, feat):
        feat = feat.mul(self.std[:-1]).add(self.mean[:-1])
        return feat

    def normalize_sdf(self, sdf):
        sdf = sdf.sub(self.mean[-1]).div(self.std[-1])
        return sdf

    def denormalize_sdf(self, sdf):
        sdf = sdf.mul(self.std[-1]).add(self.mean[-1])
        return sdf

    def normalize_re(self, re):
        re = (re - 550) / 260
        return re

    def denormalize_re(self, re):
        re = (re * 260) + 550
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

    def __getitem__(self, idx):
        if self.mode == "inference":
            # Handle inference mode with parameter sets
            re_value, Lo, Ho = self.parameter_sets[idx]
            input_data = self.inferencer.preprocess(re_value, Lo, Ho)
            return dict(
                input_feat=input_data['input_feat'].squeeze(),
                input_pos=input_data['input_pos'],
                target_feat=None,  # No target in inference mode
                output_pos=input_data['output_pos'],
                re=input_data['re'],
                name=f"param_set_{idx}"
            )

        # Training/test mode
        mesh_pos = torch.load(self.uris[idx] / "mesh_points.th", weights_only=True)
        u = torch.load(self.uris[idx] / "u.th", weights_only=True)
        v = torch.load(self.uris[idx] / "v.th", weights_only=True)
        p = torch.load(self.uris[idx] / "p.th", weights_only=True)
        target = torch.cat((u, v, p), dim=1)
        parameters = readParametersFromFileName(self.uris[idx].name, self.parameterDef)
        re = parameters['re']
        Lo = parameters['Lo']
        Ho = parameters['Ho']
        sdf = torch.load(self.uris[idx] / "mesh_sdf.th", weights_only=True).unsqueeze(1).float()

        # Normalize
        mesh_pos = self.normalize_pos(mesh_pos)
        target = self.normalize_feat(target)
        re = self.normalize_re(re)
        sdf = self.normalize_sdf(sdf)

        # Subsample outputs
        output_pos, target_feat = self.subsample(self.num_outputs, mesh_pos, target, seed=idx + 1)

        if self.use_inferencer_inputs and self.inferencer is not None:
            # Use input_pos and input_feat from the inferencer
            input_data = self.preprocess(re, Lo, Ho)
            input_pos = input_data['input_pos']
            input_feat = input_data['input_feat']
        else:
            # Subsample inputs from the current data point
            input_pos, input_feat = self.subsample(self.num_inputs, mesh_pos, sdf, seed=idx)

        return dict(
            input_feat=input_feat,
            input_pos=input_pos,
            target_feat=target_feat,
            output_pos=output_pos,
            re=re,
            name=str(self.uris[idx].name),
        )