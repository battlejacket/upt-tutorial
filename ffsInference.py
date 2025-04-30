from shapely.geometry import Point, LineString, Polygon
import numpy as np
import torch
import pygmsh
from data.ffs.readParameters import readParametersFromFileName
import time  # Add this import at the top of the file
from torch.utils.data import DataLoader
from upt.datasets.ffs_dataset import ffsDataset
from upt.collators.ffs_collator import ffsCollator


class ffsInference:
    def __init__(self, train_dataset, totalPoints, numSupernodes, model=None, device=None):
        self.model = model
        self.device = device
        self.train_dataset = train_dataset
        self.totalPoints = totalPoints
        self.numSupernodes = numSupernodes

    def infer(self, parameter_sets, batch_size=1):
        """
        Perform inference using the same DataLoader setup as training.
        """
        # Create a temporary dataset for inference
        inference_dataset = ffsDataset(
            root=self.train_dataset.root,
            num_inputs=self.train_dataset.num_inputs,
            num_outputs=self.train_dataset.num_outputs,
            mode="inference",
            crop_values=self.train_dataset.crop_values,
            inference_mode=True,
            parameter_sets=parameter_sets
        )

        # Create a DataLoader for inference
        inference_dataloader = DataLoader(
            dataset=inference_dataset,
            batch_size=batch_size,
            collate_fn=ffsCollator(num_supernodes=self.numSupernodes, deterministic=True),
        )

        all_predictions = []
        all_parameters = []
        all_points = []

        # Perform inference
        for batch in inference_dataloader:
            with torch.no_grad():
                y_hat = self.model(
                    input_feat=batch["input_feat"].to(self.device),
                    input_pos=batch["input_pos"].to(self.device),
                    supernode_idxs=batch["supernode_idxs"].to(self.device),
                    batch_idx=batch["batch_idx"].to(self.device),
                    output_pos=batch["output_pos"].to(self.device),
                    re=batch["re"].to(self.device),
                )
            all_predictions.append(y_hat.cpu())
            all_parameters.extend(batch["name"])
            all_points.append(batch["output_pos"])

        return {
            "parameters": all_parameters,
            "points": torch.cat(all_points),
            "predictions": torch.cat(all_predictions),
        }