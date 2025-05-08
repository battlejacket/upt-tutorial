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
    def __init__(self, base_dataset, numSupernodes,
                  model = None, device = None):
        
        self.model = model
        self.base_dataset = base_dataset
        # self.totalPoints = totalPoints
        self.numSupernodes = numSupernodes
        self.device = device
        # self.xMin = base_dataset.crop_values[0][0]
        # self.xMax = base_dataset.crop_values[1][0]



    def infer(self, parameter_sets, output_pos=None, batch_size=1):
        """
        Perform inference using the same DataLoader setup as training.
        """
        # Create a temporary dataset for inference
        inference_dataset = ffsDataset(
            root=self.base_dataset.root,
            num_inputs=self.base_dataset.num_inputs,
            num_outputs=self.base_dataset.num_outputs,
            mode="inference",
            crop_values=self.base_dataset.crop_values,
            parameter_sets=parameter_sets,
            useMesh=self.base_dataset.useMesh,
            meshParameters=self.base_dataset.meshParameters,
            customOutputPos=output_pos,
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
            # for key, value in batch.items():
            #     if key == 'name':
            #         continue
            #     print(key, value.shape)

            with torch.no_grad():
                y_hat = self.model(
                    input_feat=batch["input_feat"].to(self.device),
                    input_pos=batch["input_pos"].to(self.device),
                    supernode_idxs=batch["supernode_idxs"].to(self.device),
                    batch_idx=batch["batch_idx"].to(self.device),
                    output_pos=batch["output_pos"].to(self.device),
                    re=batch["re"].to(self.device),
                )
            all_predictions.append(self.base_dataset.denormalize_feat(y_hat.cpu()))
            # all_parameters.extend(batch["name"])
            all_points.append(self.base_dataset.denormalize_pos(batch["output_pos"]))


        return {
            # "parameters": all_parameters,
            "points": torch.stack(all_points),
            "predictions": torch.stack(all_predictions),
        }
    
    # def infer(self, parameter_sets, output_pos=None, batch_size=1):
    #     """
    #     Perform inference for a set of parameters using the model.
    #     Collect predictions together for easier analysis.
    #     """

    #     # Create a temporary dataset for inference
    #     inference_dataset = ffsDataset(
    #         root=self.base_dataset.root,
    #         num_inputs=self.base_dataset.num_inputs,
    #         num_outputs=self.base_dataset.num_outputs,
    #         mode="inference",
    #         crop_values=self.base_dataset.crop_values,
    #         parameter_sets=parameter_sets,
    #         useMesh=self.base_dataset.useMesh,
    #         meshParameters=self.base_dataset.meshParameters,
    #     )


    #     # Create a DataLoader for inference
    #     inference_dataloader = DataLoader(
    #         dataset=inference_dataset,
    #         batch_size=batch_size,
    #         collate_fn=ffsCollator(num_supernodes=self.numSupernodes, deterministic=True),
    #     )

    #     all_predictions = []
    #     all_parameters = []
    #     all_points = []

    #     idx = 0
    #     total_start_time = time.time()  # Start timing the entire inference process

    #     for re_value, Lo, Ho in parameter_sets:
    #         step_start_time = time.time()  # Start timing this step

    #         # Preprocessing step
    #         preprocess_start_time = time.time()
    #         batch = self.preprocess(re_value, Lo, Ho, idx)
    #         preprocess_end_time = time.time()
    #         # print(f"Preprocessing for parameters (re={re_value}, Lo={Lo}, Ho={Ho}) took {preprocess_end_time - preprocess_start_time:.4f} seconds.")

    #         # Output position setup
    #         current_output_pos = output_pos if output_pos is not None else batch['output_pos']

    #         # Inference step
    #         inference_start_time = time.time()
    #         with torch.no_grad():
    #             y_hat = self.model(
    #                 input_feat=batch['input_feat'].to(self.device),
    #                 input_pos=batch['input_pos'].to(self.device),
    #                 supernode_idxs=batch['supernode_idxs'].to(self.device),
    #                 batch_idx=batch['batch_idx'].to(self.device),
    #                 output_pos=current_output_pos.to(self.device),
    #                 re=batch['re'].to(self.device),
    #             )
    #         inference_end_time = time.time()
    #         # print(f"Inference for parameters (re={re_value}, Lo={Lo}, Ho={Ho}) took {inference_end_time - inference_start_time:.4f} seconds.")

    #         # Postprocessing step
    #         postprocess_start_time = time.time()
    #         all_points.append(self.base_dataset.denormalize_pos(current_output_pos.squeeze()))
    #         all_predictions.append(self.base_dataset.denormalize_feat(y_hat.cpu()))
    #         all_parameters.append({'re': re_value, 'Lo': Lo, 'Ho': Ho})
    #         postprocess_end_time = time.time()
    #         # print(f"Postprocessing for parameters (re={re_value}, Lo={Lo}, Ho={Ho}) took {postprocess_end_time - postprocess_start_time:.4f} seconds.")

    #         # Step timing
    #         step_end_time = time.time()
    #         # print(f"Total time for step {idx + 1} (re={re_value}, Lo={Lo}, Ho={Ho}) took {step_end_time - step_start_time:.4f} seconds.")

    #         idx += 1

    #     total_end_time = time.time()  # End timing the entire inference process
    #     # print(f"Total inference process took {total_end_time - total_start_time:.4f} seconds.")

    #     return {
    #         'parameters': all_parameters,
    #         'points': torch.stack(all_points),
    #         'predictions': torch.stack(all_predictions)
    #     }