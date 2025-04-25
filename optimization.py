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

from ffsInference import ffsInference

class ffsOptProblem(Problem):

    def __init__(self, n_var, n_obj, xl, xu, re, inferencer: ffsInference, maxDesignsPerEvaluation=100):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.gen = 0
        self.re = re
        self.maxDesignsPerEvaluation = maxDesignsPerEvaluation
        self.inferencer = inferencer

    # def readFile(self, fileDir, objective, design):
    #     file = objective + "_design_" + str(design[0]) + ".csv"
    #     with open(os.path.join(fileDir, file), "r") as datafile:
    #         data = []
    #         reader = csv.reader(datafile, delimiter=",")
    #         for row in reader:
    #             columns = [row[1]]
    #             data.append(columns)
    #         last_row = float(data[-1][0])
    #         return np.array(last_row)

    def _evaluate(self, allDesigns, out, *args, **kwargs):
        start_time = time.time()
        if self.maxDesignsPerEvaluation > allDesigns.shape[0]:
            batches = 1
        else:
            batches = int(allDesigns.shape[0]/self.maxDesignsPerEvaluation)
        
        tfFiles = glob.glob(os.path.join(self.path, "events.out.tfevents*"))

        valuesF = []
        print(f'Generation {str(self.gen)}: Evaluating {str(allDesigns.shape[0])} Designs in {str(batches)} Batches')
        # print("Generation " + str(self.gen) + ": Evaluating " + str(allDesigns.shape[0]) + " Designs in " + str(batches) + " Batches")
        
        # Create points to use for inference, one line upsteam and one line downstream of the step
        upstreamX = -3
        downstreamX = 3
        numPoints = 100
        upstreamPoints = [[y, upstreamX] for y in np.linspace(-0.5, 0.5, numPoints)]
        downstreamPoints = [[y, downstreamX] for y in np.linspace(-0.5, 0.5, numPoints)]

        # fluidParameters
        rho = 1.0
        Um = 1.0

        for designs in np.array_split(ary=allDesigns, indices_or_sections=batches):
            # Inference
            # Create a column of c values with the same number of rows as a
            re_column = np.full((allDesigns.shape[0], 1), self.re)

            # Concatenate along the second axis (columns)
            paramer_sets = np.concatenate((re_column, allDesigns), axis=1)

            print(paramer_sets)

            upstreamResults = self.inferencer.infer(parameter_sets=designs, output_pos=upstreamPoints)
            downstreamResults = self.inferencer.infer(parameter_sets=designs, output_pos=downstreamPoints)

            valuesF.append(2*(upstreamResults[2]-downstreamResults[2])/(rho*Um**2))


        out["F"] = np.array(valuesF)
        self.gen += 1
        elapsed_time = time.time() - start_time
        print("Evaluation time: ", elapsed_time)


