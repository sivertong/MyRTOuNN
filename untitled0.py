# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 21:13:00 2023

@author: long
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import time
import matplotlib.pyplot as plt
from TopologyOptimizer1 import TopologyOptimizer


nelx = 160; # number of FE elements along X 观测点80*80，有限元160*160
nely = 160; # number of FE elements along Y
elemSize = np.array([1.0,1.0]);
mesh = {'nelx':nelx, 'nely':nely, 'elemSize':elemSize};

matProp = {'E':1.0, 'nu':0.3}; # Structural
matProp['penal'] = 3; # SIMP penalization constant, starting value

exampleName = 'TipCantilever'
physics = 'Structural' #可以选择'Structural'或'Thermal'
ndof = 2*(nelx+1)*(nely+1);
force = np.zeros((ndof,1))
dofs=np.arange(ndof);
fixed = dofs[0:2*(nely+1):1];
force[2*(nelx+1)*(nely+1)-(nely+1), 0 ] = -1;
symXAxis = {'isOn':False, 'midPt':0.5*nely};
symYAxis = {'isOn':False, 'midPt':0.5*nelx};
bc = {'exampleName':exampleName, 'physics':physics, \
      'force':force, 'fixed':fixed, 'symXAxis':symXAxis, 'symYAxis':symYAxis };

# For more BCs see examples.py

nnSettings = {'numLayers':3, 'numNeuronsPerLyr':20 }

densityProjection = {'isOn':True, 'sharpness':8};
desiredVolumeFraction = 0.5;

minEpochs = 150; # minimum number of iterations
maxEpochs = 500; # Max number of iterations
plt.close('all');
overrideGPU = False
start = time.perf_counter()
# topOpt = TopologyOptimizer(mesh, matProp, bc, nnSettings, \
#                   desiredVolumeFraction, densityProjection, overrideGPU);
topOpt = TopologyOptimizer(mesh, matProp, bc, nnSettings, \
                            desiredVolumeFraction, densityProjection, overrideGPU)
topOpt.OptimizeDesign(maxEpochs,minEpochs);
print("Time taken (secs): {:.2F}".format( time.perf_counter() - start))

