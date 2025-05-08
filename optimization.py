import numpy as np
import dill
import os, glob, io, time
from os import listdir
import csv
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.visualization.scatter import Scatter
import contextlib
from multiprocessing import Process
from pymoo.termination.default import DefaultMultiObjectiveTermination
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
        # valuesF = []
        # print(f'Generation {str(self.gen)}: Evaluating {str(allDesigns.shape[0])} Designs in {str(batches)} Batches')
        # print("Generation " + str(self.gen) + ": Evaluating " + str(allDesigns.shape[0]) + " Designs in " + str(batches) + " Batches")
        
        # Create points to use for inference, one line upsteam and one line downstream of the step
        upstreamX = -3
        downstreamX = 3
        numPoints = 5
        upstreamPoints = torch.tensor([[upstreamX, y] for y in np.linspace(-0.5, 0.5, numPoints)], dtype=torch.float32)
        downstreamPoints = torch.tensor([[downstreamX, y] for y in np.linspace(-0.5, 0.5, numPoints)], dtype=torch.float32)
        points = torch.cat((upstreamPoints, downstreamPoints), dim=0) #.unsqueeze(0)

        # points = self.inferencer.train_dataset.normalize_pos(points)

        # fluidParameters
        rho = 1.0
        Um = 1.0

        # for designs in np.array_split(ary=allDesigns, indices_or_sections=batches):
        designs = allDesigns
            # Inference
            # Create a column of c values with the same number of rows as a
        re_column = np.full((designs.shape[0], 1), self.re)

        # Concatenate along the second axis (columns)
        parameter_sets = np.concatenate((re_column, designs), axis=1)

        # print("Parameter sets shape: ", parameter_sets.shape)
        # print("Parameter sets: ", parameter_sets)
        
        results = self.inferencer.infer(parameter_sets=parameter_sets, output_pos=points)
        prediction = results['predictions']
        
        upstreamPressures = prediction[:, :, 2][:,:numPoints]
        downstreamPressures = prediction[:, :, 2][:,numPoints:]

        upstreamPressure = upstreamPressures.mean(dim=1)
        downstreamPressure = downstreamPressures.mean(dim=1)
        
        dCp =  2*(upstreamPressure-downstreamPressure)/(rho*Um**2)

        # print("dCp shape: ", dCp.shape)
        # print("dCp: ", dCp)

        # valuesF.append(dCp.detach().cpu().numpy())

        out["F"] = dCp.detach().cpu().numpy() #np.array(valuesF)
        self.gen += 1
        elapsed_time = time.time() - start_time
        print("Evaluation time: ", elapsed_time)


