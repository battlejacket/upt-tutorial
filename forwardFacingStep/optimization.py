import numpy as np
import os, glob, io, time
from os import listdir
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import contextlib
import torch
from ffsInference import ffsInference

class deltaCp():
    def __init__(self, rho = 1.0, Um = 1.0, upstreamX = -4, downstreamX = 4, numPoints = 100):
        self.upstreamX = upstreamX
        self.downstreamX = downstreamX
        self.numPoints = numPoints
        self.rho = rho
        self.Um = Um

    def outputPositions(self):
        # Create points to use for inference, one line upstream and one line downstream of the step
        upstreamPoints = torch.tensor([[self.upstreamX, y] for y in np.linspace(-0.5, 0.5, self.numPoints)], dtype=torch.float32)
        downstreamPoints = torch.tensor([[self.downstreamX, y] for y in np.linspace(-0.5, 0.5, self.numPoints)], dtype=torch.float32)
        points = torch.cat((upstreamPoints, downstreamPoints), dim=0)
        return points
    
    def calculateDeltaCp(self, predictions):

        # Reshape predictions to [num_designs, num_points, 3]
        predictions = predictions.reshape(int(predictions.shape[0]/(self.numPoints*2)), self.numPoints*2, -1)

        # Extract pressures
        pressures = predictions[:, :, 2]  # Extract the pressure (p) component

        # Split into upstream and downstream pressures
        upstreamPressures = pressures[:, :self.numPoints]
        downstreamPressures = pressures[:, self.numPoints:]

        # Calculate average pressures
        upstreamPressure = upstreamPressures.mean(dim=1)
        downstreamPressure = downstreamPressures.mean(dim=1)

        # Calculate dCp
        dCp = 2 * (upstreamPressure - downstreamPressure) / (self.rho * self.Um**2)
        return dCp

class ffsOptProblem(Problem):
    def __init__(self, n_var, n_obj, xl, xu, inferencer: ffsInference, re=500.0, maxDesignsPerEvaluation=100):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.gen = 0
        self.re = re
        self.maxDesignsPerEvaluation = maxDesignsPerEvaluation
        self.inferencer = inferencer
        self.deltaCp = deltaCp()

    def _evaluate(self, allDesigns, out, *args, **kwargs):
        # start_time = time.time(s)

        # Create points to use for inference, one line upstream and one line downstream of the step
        output_pos = self.deltaCp.outputPositions()

        # Prepare parameter sets
        re_column = np.full((allDesigns.shape[0], 1), self.re)
        parameter_sets = np.concatenate((re_column, allDesigns), axis=1)

        # Perform inference
        results = self.inferencer.infer(parameter_sets=parameter_sets, output_pos=output_pos, batch_size=self.maxDesignsPerEvaluation)
        predictions = results['predictions']  # Shape: [num_designs * num_points, 3]

        dCp = self.deltaCp.calculateDeltaCp(predictions=predictions)

        # Store results
        out["F"] = dCp.detach().cpu().numpy()
        self.gen += 1
        # elapsed_time = time.time() - start_time
        # print(f"Evaluation time: {elapsed_time:.2f} seconds")
