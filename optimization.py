import numpy as np
import os, glob, io, time
from os import listdir
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import contextlib
import torch
from ffsInference import ffsInference

class ffsOptProblem(Problem):
    def __init__(self, n_var, n_obj, xl, xu, inferencer: ffsInference, re=500.0, maxDesignsPerEvaluation=100):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.gen = 0
        self.re = re
        self.maxDesignsPerEvaluation = maxDesignsPerEvaluation
        self.inferencer = inferencer

    def _evaluate(self, allDesigns, out, *args, **kwargs):
        start_time = time.time()

        # Create points to use for inference, one line upstream and one line downstream of the step
        upstreamX = -4
        downstreamX = 4
        numPoints = 100
        upstreamPoints = torch.tensor([[upstreamX, y] for y in np.linspace(-0.5, 0.5, numPoints)], dtype=torch.float32)
        downstreamPoints = torch.tensor([[downstreamX, y] for y in np.linspace(-0.5, 0.5, numPoints)], dtype=torch.float32)
        points = torch.cat((upstreamPoints, downstreamPoints), dim=0)

        # Prepare parameter sets
        re_column = np.full((allDesigns.shape[0], 1), self.re)
        parameter_sets = np.concatenate((re_column, allDesigns), axis=1)

        # Perform inference
        results = self.inferencer.infer(parameter_sets=parameter_sets, output_pos=points, batch_size=self.maxDesignsPerEvaluation)
        predictions = results['predictions']  # Shape: [num_designs * num_points, 3]

        # Reshape predictions to [num_designs, num_points, 3]
        num_designs = allDesigns.shape[0]
        num_total_points = points.shape[0]
        predictions = predictions.reshape(num_designs, num_total_points, -1)

        # Extract pressures
        pressures = predictions[:, :, 2]  # Extract the pressure (p) component

        # Split into upstream and downstream pressures
        upstreamPressures = pressures[:, :numPoints]
        downstreamPressures = pressures[:, numPoints:]

        # Calculate average pressures
        upstreamPressure = upstreamPressures.mean(dim=1)
        downstreamPressure = downstreamPressures.mean(dim=1)

        # Calculate dCp
        rho = 1.0
        Um = 1.0
        dCp = 2 * (upstreamPressure - downstreamPressure) / (rho * Um**2)

        # Store results
        out["F"] = dCp.detach().cpu().numpy()
        self.gen += 1
        elapsed_time = time.time() - start_time
        print(f"Evaluation time: {elapsed_time:.2f} seconds")
