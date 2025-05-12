from shapely.geometry import Point, LineString, Polygon
import numpy as np
import torch
import pygmsh
from data.ffs.readParameters import readParametersFromFileName
import time  # Add this import at the top of the file
from torch.utils.data import DataLoader
from upt.datasets.ffs_dataset import ffsDataset
from upt.collators.ffs_collator import ffsCollator
import copy

class ffsInference:
    def __init__(self, base_dataset, numSupernodes, model=None, device=None):
        self.model = model
        self.base_dataset = base_dataset
        self.numSupernodes = numSupernodes
        self.device = device
        self.inference_dataset = copy.deepcopy(base_dataset)

    def infer(self, parameter_sets, output_pos=None, batch_size=1):
        """
        Perform inference using the same DataLoader setup as training.
        """
        self.inference_dataset.setInferenceMode(parameter_sets)
        self.inference_dataset.customOutputPos = output_pos

        # Create a DataLoader for inference
        inference_dataloader = DataLoader(
            dataset=self.inference_dataset,
            batch_size=batch_size,
            collate_fn=ffsCollator(num_supernodes=self.numSupernodes, deterministic=True),
        )

        all_predictions = []
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
            # all_predictions.append(y_hat.cpu())
            # all_points.append(batch["output_pos"])
            all_predictions.append(self.base_dataset.denormalize_feat(y_hat.cpu()))
            all_points.append(self.base_dataset.denormalize_pos(batch["output_pos"]))

        # Concatenate predictions and points across batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_points = torch.cat(all_points, dim=0)

        return {
            "points": all_points,
            "predictions": all_predictions,
        }

# class ffsInference:
#     def __init__(self, base_dataset, numSupernodes,
#                   model = None, device = None):
        
#         self.model = model
#         self.base_dataset = base_dataset
#         # self.totalPoints = totalPoints
#         self.numSupernodes = numSupernodes
#         self.device = device
#         # self.xMin = base_dataset.crop_values[0][0]
#         # self.xMax = base_dataset.crop_values[1][0]

#         self.inference_dataset = copy.deepcopy(base_dataset)


#     def inferOld(self, parameter_sets, output_pos=None, batch_size=1):
#         """
#         Perform inference using the same DataLoader setup as training.
#         """
#         # Create a temporary dataset for inference
#         # self.inference_dataset = ffsDataset(
#         #     root=self.base_dataset.root,
#         #     num_inputs=self.base_dataset.num_inputs,
#         #     num_outputs=self.base_dataset.num_outputs,
#         #     mode="inference",
#         #     crop_values=self.base_dataset.crop_values,
#         #     parameter_sets=parameter_sets,
#         #     useMesh=self.base_dataset.useMesh,
#         #     meshParameters=self.base_dataset.meshParameters,
#         #     customOutputPos=output_pos,
#         # )
        
#         # self.inference_dataset = copy.deepcopy( self.base_dataset)
#         self.inference_dataset.setInferenceMode(parameter_sets)
#         self.inference_dataset.customOutputPos=output_pos
        
#         # Create a DataLoader for inference
#         inference_dataloader = DataLoader(
#             dataset=self.inference_dataset,
#             batch_size=batch_size,
#             collate_fn=ffsCollator(num_supernodes=self.numSupernodes, deterministic=True),
#         )

#         all_predictions = []
#         all_parameters = []
#         all_points = []

#         # Perform inference
#         for batch in inference_dataloader:
#             # for key, value in batch.items():
#             #     if key == 'name':
#             #         continue
#             #     print(key, value.shape)

#             with torch.no_grad():
#                 y_hat = self.model(
#                     input_feat=batch["input_feat"].to(self.device),
#                     input_pos=batch["input_pos"].to(self.device),
#                     supernode_idxs=batch["supernode_idxs"].to(self.device),
#                     batch_idx=batch["batch_idx"].to(self.device),
#                     output_pos=batch["output_pos"].to(self.device),
#                     re=batch["re"].to(self.device),
#                 )
#             all_predictions.append(self.base_dataset.denormalize_feat(y_hat.cpu()))
#             # all_parameters.extend(batch["name"])
#             all_points.append(self.base_dataset.denormalize_pos(batch["output_pos"]))


#         return {
#             # "parameters": all_parameters,
#             "points": torch.stack(all_points),
#             "predictions": torch.stack(all_predictions),
#         }
